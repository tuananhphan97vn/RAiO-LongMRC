

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import math 


def get_spare_embed(sent_hidden_state , batch_bound_sents):
	#this function is used to remove padding vector from tensor. 
	#sent hidden state: (bs , max sent , sent dim ) now will be converted into form of list of tensor (N_sent , sent dim). In this, N_sent can be different between elements 
	# Second list is list of sentences in each passages. Each list can have different length 
	# Each element in list is tensor. This tensor has shape (N_words , word dim ). N_words is number of word appeared in particular sentence
	# To create the aforementioned object, we use batch bound sent  to extract number of words in sentence as well as number of sentences in passage 
	#batch bound sent is list , each element in list is 2-element list. It express start token and end token in certain sentence
	sent_result = []
	for i in range(len(batch_bound_sents)):
		sent_result.append(sent_hidden_state[i][:len(batch_bound_sents[i])]) #only acquire real sentence 
	return sent_result

class SelfAttentionBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim 
		self.W1 = nn.Linear(self.sent_dim , self.sent_dim) # query matrix  
		self.W2 = nn.Linear(self.sent_dim , self.sent_dim) # value matrix 

	def forward(self, list_sentence_embedding, ):
		#sentence embedding is list, one element in list is tensor, each tensor shape (number sentence , sent dim )
		result = [] 
		for i in range(len(list_sentence_embedding)):
			sent_embed = list_sentence_embedding[i]  #shape (N_sent , sent dim )
			sent_query = self.W1(sent_embed) #shape (N_sent , sent dim)
			z = 1. / math.sqrt(self.sent_dim) * ( torch.matmul( sent_query , torch.transpose(sent_query , 0 , 1 )) ) # shape (N_sent , N_sent )
			attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_sent)
			sent_value = self.W2(sent_embed) #shape (N_sent , sent dim )
			sent_hidden_state = torch.matmul(attention_matrix , sent_value) #shape (N_sent , sent dim )
			result.append(sent_hidden_state) #list of sent hidden, each sent hidden has shape (N_sent, sent dim )
		return torch.stack(result) #shape (bs , N_sents , sent dim )

class WordSentenceIntegrateBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim 
		self.W1 = nn.Linear(2* self.sent_dim , self.sent_dim) # word concate with sent, then pass through linear layer to 

	def forward(self, words_emb , sents_emb, batch_bound_sents ):
		#word emb shape (bs , seq len , word dim)
		#sents emb (bs , max sent in batch , sent dim )
		#batch bound sents is list, which consist of 2-element sublist, [start_po, end_po] of each sentence in passage 
		sents_emb  = get_spare_embed(sents_emb , batch_bound_sents) 
		N_sample =  words_emb.shape[0] 
		sents_state_flat = [] 
		for i in range(N_sample):
			boundaries = batch_bound_sents[i]
			sents = sents_emb[i]  #sents shape (N_sent , sent dim )
			start_bound = boundaries[0][0] #start of first sent due to we do not consider question
			end_bound = boundaries[-1][1] #end of last sent 
			list_emb = []
			for j in range(sents.shape[0]):
				[start , end] = boundaries[j]
				repeat_sent_emb = sents[j , :].repeat(end-start+1 , 1) # repeat vector (hidden dim) ---> 2-D tensor (N_word , hidden dim )
				list_emb.append(repeat_sent_emb)
			sent_emb = torch.cat(list_emb) #shape (text len , hidden size)
			padd_sent_emd = torch.cat ([torch.zeros(start_bound, words_emb.shape[2]).to(torch.device('cuda:0')), sent_emb , torch.zeros(words_emb.shape[1] - 1 - end_bound , words_emb.shape[2]).to(torch.device('cuda:0'))]) #shape (seq len , hidden size )
			sents_state_flat.append(padd_sent_emd)
		sents_state_flat = torch.stack(sents_state_flat) #shape (bs , seq len , hidden size )

		# word_sent_embed = last_hidden_state + sents_state_flat #shape (bs , seq len , hidden size ) #this is in case we use add operator 
		word_sent_embed = torch.cat( (words_emb , sents_state_flat) , dim = 2) #shape (bs , seq len , 2* hidden size )
		word_sent_embed = torch.nn.functional.relu(self.W1(word_sent_embed)) #shape (bs , seq len , hidden size)
		return word_sent_embed

class QuestionAwareBlock(nn.Module):

	def __init__(self, sent_dim):
		super().__init__()
		self.sent_dim = sent_dim 
		self.W1 = nn.Linear(self.sent_dim , self.sent_dim) # query matrix  
		self.W2 = nn.Linear(self.sent_dim , self.sent_dim) # value matrix 		

	def forward(self , ques_emb , sents_emb ):
		# ques_emb shape (bs , sent dim ) 
		# sents emd: (bs , max sent in batch , sent dim )

		query_question = ques_emb.unsqueeze(1) #shape (bs , 1 , sent dim )
		query_question = self.W1(query_question) #shape (bs , 1 , sent dim )
		key_sents = self.W1(sents_emb) #shape (bs , max_sent_in_batch , sent_dim )
		z  =  torch.bmm(query_question , torch.transpose ( key_sents , 1 , 2))  #shape (bs , 1, max_sent)
		z = z / math.sqrt(self.sent_dim)
		atten_score = torch.nn.functional.softmax(z , dim = 2) #shape (bs , 1 , max sent)
		atten_score = torch.squeeze(atten_score ,  dim = 1) #shape (bs , max sent)
		
		val_sents = self.W2(sents_emb) #shape (bs , max sent , sent dim )
		val_ques = self.W2(ques_emb) #shape (bs ,  sent dim)
		val_ques_duplicate = val_ques.unsqueeze(1).repeat(1 , sents_emb.shape[1] , 1) #shape (bs , max sent , sent dim )

		sent_hiddens = torch.tanh( atten_score.unsqueeze(2) * val_ques_duplicate + (1 - atten_score).unsqueeze(2) * val_sents ) #shape (bs , max sent , sent dim )
		return sent_hiddens   

class FeedForwardBlock(nn.Module):

	def __init__(self, input_dim , output_dim):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim  = output_dim
		self.W1 = nn.Linear(self.input_dim , self.output_dim)
		self.W2 = nn.Linear(self.output_dim , self.input_dim)

	def forward(self, sents_emb):
		#sents emb shape (bs , N_sent , sent dim )
		x = torch.nn.functional.relu(self.W1(sents_emb)) #shape (bs , N-sent , output dim )
		x = torch.nn.functional.relu(self.W2(x)) #shape (bs , N_sent , input dim )
		return x 

class ResidualAndNorm(nn.Module):

	def __init__(self , sent_dim):
		super().__init__()
		self.sent_dim = sent_dim
		self.layer_norm = nn.LayerNorm(self.sent_dim )
	
	def forward(self, residual, output ):
		#residual, output have shape (bs , max sent , sent dim )
		output = self.layer_norm(output + residual)
		return output

class SentenceEncodeBlock(nn.Module):

	def __init__(self , sent_dim):
		super().__init__()
		self.sent_dim = sent_dim 
		#initialize three important components 
		self.question_aware_block = QuestionAwareBlock(self.sent_dim)
		self.self_attention_block = SelfAttentionBlock(self.sent_dim)
		self.feed_forward_block = FeedForwardBlock(self.sent_dim , self.sent_dim)
		self.residual_norm= ResidualAndNorm(self.sent_dim)
		self.word_sent = WordSentenceIntegrateBlock(self.sent_dim)

	def forward(self, ques_emb , sents_emb): 
		#ques emb shape (bs , sent dim )
		#sents emb shape (bs , max sent in batch , sent dim )
		residual = sents_emb
		out_sent  = self.question_aware_block(ques_emb , sents_emb )
		sents_emb = self.residual_norm(residual, out_sent)

		residual = sents_emb
		out_sent = self.self_attention_block(sents_emb)
		sents_emb = self.residual_norm(residual , out_sent)

		residual = sents_emb
		out_sent = self.feed_forward_block(sents_emb)
		sents_emb = self.residual_norm(residual , out_sent)

		return sents_emb 

class SentenceEncoder(nn.Module):

	def __init__(self, n_block , sent_dim , device):
		#one sentence encoder consist of multi sentence encoder block 
		super().__init__()
		self.device = device
		self.n_block = n_block
		self.sent_dim = sent_dim 
		self.list_sent_encoder_block =  nn.ModuleList([ SentenceEncodeBlock(self.sent_dim)  for i in range(self.n_block)])
		self.list_word_sent_integrate = nn.ModuleList([ WordSentenceIntegrateBlock(self.sent_dim) for i in range(self.n_block)])

	def forward(self, ques_embed ,sents_emb, last_hidden_state , batch_bound_sents):
		word_embs = last_hidden_state
		for i in range(self.n_block):
			sents_emb = self.list_sent_encoder_block[i](ques_embed , sents_emb) #update sents embedding after each iteration
			word_embs = self.list_word_sent_integrate[i](word_embs , sents_emb , batch_bound_sents) #using new word embedding using aformentioned sents embedding 
		return word_embs
