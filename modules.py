import chunk
from sre_constants import NOT_LITERAL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import math
from torch.nn.utils.rnn import pad_sequence
import time
from collections import Counter

class ResidualAndNorm(nn.Module):

	def __init__(self , sent_dim):
		super().__init__()
		self.sent_dim = sent_dim
		self.layer_norm = nn.LayerNorm(self.sent_dim)
	
	def forward(self, origin , out_put ):
		#residual, output have shape (bs , max sent , sent dim )
		return self.layer_norm(origin + out_put)

def pool_sequential_embed(embed , start , end , method):
	if method =='mean':
		sub_matrix = embed[start:end+1 , :]
		return torch.mean(sub_matrix , axis = 0 )

class SelfAttentionBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim
		self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
		self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

	def forward(self, sent_embed):

		sent_query = self.W_q(sent_embed) #shape (N_sent , sent dim)
		sent_key = self.W_k(sent_embed)
		sent_value = self.W_v(sent_embed) #shape (N_sent , sent dim )

		z = 1. / math.sqrt(self.sent_dim) * ( torch.matmul( sent_query , torch.transpose(sent_key , 0 , 1 )) ) # shape (N_sent , N_sent )
		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_sent)
		result = torch.matmul(attention_matrix , sent_value) #shape (N_sent , sent dim )
		return torch.nn.functional.relu(result) #shape (N_sent, sent dim)

class MultiBlockSelfAttention(nn.Module):

	def __init__(self , sent_dim , n_iter) -> None:
		super().__init__()
		self.sent_dim  = sent_dim 
		self.n_iter = n_iter 
		self.multi_selfattentionblock = nn.ModuleList([ SelfAttentionBlock(self.sent_dim) for i in range(self.n_iter) ])
		self.multi_layernorm = nn.ModuleList([ nn.LayerNorm(self.sent_dim) for i in range(self.n_iter)])	

	def forward(self , sents_emb):
		#sents emb shape (N_sent , sent dim)
		sents_hidden = sents_emb
		for i in range(self.n_iter):
			sents_hidden = self.multi_layernorm[i](sents_hidden + self.multi_selfattentionblock[i](sents_hidden) )
		return sents_hidden

class SelfAttentionSpareBlock(nn.Module):

	def __init__(self, sent_dim , device ):
		super().__init__()
		self.sent_dim = sent_dim
		self.device = device 
		self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
		self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

	def forward(self, sent_embed , adj_tensor):

		sent_query = self.W_q(sent_embed) #shape (N_sent , sent dim)
		sent_key = self.W_k(sent_embed)
		sent_value = self.W_v(sent_embed) #shape (N_sent , sent dim )

		z = 1. / math.sqrt(self.sent_dim) * ( torch.matmul( sent_query , torch.transpose(sent_key , 0 , 1 )) ) # shape (N_sent , N_sent )
		spare_z = z * adj_tensor # shape (N_sent , N_sent ) #spare tensor with many value equal 0 
		spare_z = torch.where(spare_z == 0. , torch.tensor(-10000, dtype = spare_z.dtype).to(self.device) , spare_z) # if spare_z == 0 , -- > -inf -- > softmax = 0  

		attention_matrix = torch.nn.functional.softmax(spare_z , dim = 1) #shape (N_sent , N_sent)
		result = torch.matmul(attention_matrix , sent_value) #shape (N_sent , sent dim) 
		
		return torch.nn.functional.relu(result) #shape (N_sent, sent dim)

class CrossAttentionBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim
		self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
		self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

	def forward(self, ques_emb , sent_embed):

		sent_query = self.W_q(sent_embed) #shape (N_sent , sent dim)
		ques_key = self.W_k(ques_emb) #shape (N word in question , question dim = sent dim)
		ques_val = self.W_v(ques_emb) #shape (N_word , sent dim )

		z = 1. / math.sqrt(self.sent_dim) * ( torch.matmul( sent_query , torch.transpose(ques_key , 0 , 1 )) ) # shape (N_sent , N_word )
		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_word )
		result = torch.matmul(attention_matrix , ques_val) #shape (N_sent , sent dim )

		return torch.nn.functional.relu(result) #shape (N_sent, sent dim)


class Bottom_up_embed(nn.Module):

	def __init__(self , word_dim , sent_dim, n_block_reasoning , chunk_dim, device):
		super().__init__()

		self.word_dim = word_dim
		self.sent_dim = sent_dim
		self.chunk_dim = chunk_dim
		self.n_block_reasoning = n_block_reasoning
		self.device = device

		self.W_chunk_pool = nn.Linear(self.sent_dim , self.chunk_dim)
		self.que_sent_word_reasoning = 	nn.ModuleList([QuestionSentenceWordReasoning(self.word_dim, self.device)  for i in range(self.n_block_reasoning)])	

	def extract_context_question(self, last_hidden_state , bound_passages):

		context , ques = [] , []  
		for i in range(len(last_hidden_state)):
			start , end  = bound_passages[i][0] , bound_passages[i][1]
			context.append(last_hidden_state[i][start : end + 1 , : ]) #shape (N_word in passage , word dim )
			ques.append(last_hidden_state[i][1:start - 2 , : ]) #shape (N word in question , word dim )
		
		context = torch.cat( context , dim = 0 ) #shape (N word in doc , word dim )
		ques = torch.stack(ques) #shape (N_chunk , N_word in question , word dim)

		return context , ques

	def extract_original_word(self, list_word_hidden, bound_passages):
		result = [] 
		for i in range(len(list_word_hidden)):
			words_hidden = list_word_hidden[i]
			result.append(words_hidden[bound_passages[i][0] : bound_passages[i][1] + 1 ])
		return torch.cat(result , dim = 0)

	def forward(self, last_hidden_state_passages,  bound_passages , sent2subword  , subword2sents) :

		# words_hiddens , ques_hidden = self.extract_context_question(last_hidden_state_passages , bound_passages) # (N word in context , word dim)
		list_word_hid = last_hidden_state_passages
		for i in range(self.n_block_reasoning):
			list_word_hid = self.que_sent_word_reasoning[i]( list_word_hid , bound_passages , subword2sents) 

		cls = [list_word_hid[i][0,:] for i in range(len(list_word_hid))] #list of element , each element shape (word im)
		cls = torch.stack(cls ) 
		cls = torch.mean(cls , dim = 0 ) 

		words_hidden = self.extract_original_word(list_word_hid , bound_passages)

		result = torch.cat( (cls.unsqueeze(0) , words_hidden) , dim = 0 )  #shape (N_word in doc + 1 , word dim)
		return result

class QuesWordSentAttention(nn.Module):

	def __init__(self , sent_dim):
		super().__init__()
		self.sent_dim = sent_dim
		# self.ques_word_sent_attention = SelfAttentionBlock(self.sent_dim) 
		self.n_iter = 2 
		self.ques_word = MultiBlockSelfAttention(self.sent_dim , self.n_iter)

	def forward(self, ques_hidden , ques_share, words_hiddens , bound_passages , subword2sents , alpha_sent ):

		sents_ids = sum(subword2sents , [] ) #concate subword2sents
		uni_sent = list(set(sents_ids))
		flat_alpha_sent = [] 
		for i in range(len(uni_sent)):
			flat_alpha_sent += [alpha_sent[i]]* sents_ids.count(i)
		flat_alpha_sent = torch.tensor(flat_alpha_sent)

		len_chunk = [] 
		for i in range(len(bound_passages)):
			start , end = bound_passages[i][0] , bound_passages[i][1]
			len_chunk.append(end - start + 1 ) 
		if sum(len_chunk) != words_hiddens.shape[0]:
			print('error len of list ' )
			return 0 
		chunk_embeds = torch.split(words_hiddens , len_chunk) #list of tensor, each tensor shape (N word in chunk , word dim)
		chunk_alpha = torch.split(flat_alpha_sent , len_chunk)

		ques_share = ques_share.repeat(ques_hidden.shape[0] , 1 , 1) #shape (n chunk , ques length , word dim)

		ques_hidden = ques_hidden + ques_share 

		ques_ori = ques_hidden 
		words_ori = words_hiddens 
		new_ques , add_words = [] ,  []
		length_ques = ques_hidden.shape[1]
		for i in range(len(chunk_embeds)):

			chunk_emb = chunk_embeds[i] # (chunk length , word dim)
			alpha = torch.unsqueeze(chunk_alpha[i] , 1).to(torch.device('cuda:0')) #shape (chunk length , 1)
			chunk_emb = chunk_emb * alpha #alpha now is considered as filter information
			# print('chunk emb ', chunk_emb)
			ques_word_hidden = torch.cat( (ques_ori[i , : ] , chunk_emb ) , dim = 0 ) # (que length + chunk length , word dim )
			ques_word_hidden = self.ques_word(ques_word_hidden) # (que length + chunk length , word dim ) #multi self attention block 

			new_ques.append( ques_ori[i, : ] + ques_word_hidden[: length_ques  , : ])# (que length , word dim)
			add_words.append(ques_word_hidden[length_ques: , : ]) #  (chunk length , word dim )

		ques_update = torch.stack(new_ques) #shape (N_chunk, ques length , word dim)

		add_words = torch.cat(add_words , dim = 0 ) # ( doc length , word dim)
		words_update = words_ori + add_words # ( doc length , word dim)
		return ques_update , words_update 

class Sent2SentBlock(nn.Module):

	def __init__(self , sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim
		self.sent2sent = SelfAttentionBlock(self.sent_dim )
		self.residual_norm = ResidualAndNorm(self.sent_dim)

	def forward(self, list_sent_hid ):
		#sents emb shape (N-sent , sent dim )
		list_num_sent = [t.shape[0] for t in list_sent_hid]
		sents_emb = torch.cat(list_sent_hid , dim = 0 ) # ( num sent in doc , sent dim)
		original = sents_emb 
		result = self.sent2sent(sents_emb) #shape (N_sent , sent dim)
		result = self.residual_norm(original , result) #N sent, sent dim 
		result = torch.split(result , list_num_sent) # list of tensor , tenshor shape (num sent in each chunk, sent dim )
		return result 


class Word2SentBlock(nn.Module):

	def __init__(self, sent_dim , device):
		#one sentence encoder consist of multi sentence encoder block
		super().__init__()
		self.device = device
		self.sent_dim = sent_dim
		self.word2sent = nn.Linear(self.sent_dim , self.sent_dim)

	def extract_context(self, list_word_hid , bound_passages):
		context  = [] 
		for i in range(len(list_word_hid)):
			start , end  = bound_passages[i][0] , bound_passages[i][1]
			context.append(list_word_hid[i][start : end + 1 , : ]) #shape (N_word in passage , word dim )		
		context = torch.cat( context , dim = 0 ) #shape (N word in doc , word dim )
		return context 

	def pool_sents(self, list_word_hid , bound_passages, subword2sents):

		ques_length = bound_passages[0][0] # ques length + number special words such as cls/ sep
		list_sent_hid = [] 
		for i in range(len(list_word_hid)):
			word_hid = list_word_hid[i][bound_passages[i][0] : bound_passages[i][1] + 1 , : ] # context length , word dim
			subword2sent = subword2sents[i]
			counter = Counter(subword2sent) # dict {key = sentid ,value = sent length }. But, not be sorted by key
			sent_len = [ t[1] for t in sorted(counter.items()) ]
			sents = torch.split(word_hid, sent_len) # list of tensor shape (sent length , word dim)]
			list_sent_hid.append(sents)
		
		sent_hids = [] 
		for i in range(len(list_sent_hid)):
			_ = [] 
			for j in range(len(list_sent_hid[i])):
				_.append(torch.mean(list_sent_hid[i][j] ,  dim = 0 ))
			_ = torch.stack(_ , dim = 0 ) 
			sent_hids.append(_)
		return sent_hids  

	def forward(self, words_emb, bound_passages , sent2subword):
		#words emb: tensor shape (N-word , word dim )
		#sent 2 subword is dict, key = sent id, value = order of word that appear in this sentence
		output = self.pool_sents(words_emb , bound_passages , sent2subword) #shape (N_sent , sent dim)
		return output
class QuesSent(nn.Module):

	def __init__(self , sent_dim  ):
		super().__init__()
		self.sent_dim = sent_dim
		self.n_iter = 2
		self.ques_sent_att = MultiBlockSelfAttention(self.sent_dim , self.n_iter)

	def forward(self, ques_emb , sents_hidden):
		#ques emb shape (N_chunk , n_word , word dim )
		#sents emb shape ( N_sent, hidden dim )
		ques_len = ques_emb.shape[1]
		ques_tensor = torch.mean(ques_emb , dim = 0) # (n_word , word dim)
		# print(ques_tensor.shape)

		ques_sent_hid = torch.cat( (ques_tensor , sents_hidden) , dim = 0 ) # (N_sent + N_word in question , word dim)
		ques_sent_hid = self.ques_sent_att(ques_sent_hid) # (N_sent + N_word , word dim)
		return ques_sent_hid[ : ques_len  , :]  ,  ques_sent_hid[ques_len : , : ]

class QuesSentAlpha(nn.Module):

	def __init__(self , sent_dim  ):
		super().__init__()
		self.sent_dim = sent_dim
		self.ques_linear = nn.Linear(self.sent_dim , self.sent_dim)
		self.sent_linear = nn.Linear(self.sent_dim , self.sent_dim)

	def forward(self, ques_share , sents_hidden):
		ques_vector= torch.unsqueeze(torch.mean(ques_share , dim = 0 ) , 0 ) # (1 , word dim)
		ques_query = self.ques_linear(ques_vector) 
		sent_key = self.sent_linear(sents_hidden)  #shape (n_sent , sent dim)
		z = torch.mm(ques_query , sent_key.T) #shape (1, n_sent)
		z = torch.nn.functional.softmax(z , dim = 1) #shape (1 , n _sent)
		return z.squeeze(0) # (N_sent)
