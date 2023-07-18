

from math import trunc
from os import truncate
from tracemalloc import start
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from SentenceEncoder import SentenceEncoder 
#from ClassifierLayer import Classifier
from modules import Bottom_up_embed

def get_batch_from_feature(features , batch_index): 
	all_input_ids = torch.tensor([features[t].input_ids for t in batch_index], dtype=torch.long)
	all_attention_masks = torch.tensor([features[t].attention_mask for t in batch_index], dtype=torch.long)
	all_token_type_ids = torch.tensor([features[t].token_type_ids for t in batch_index], dtype=torch.long)
	all_cls_index = torch.tensor([features[t].cls_index for t in batch_index], dtype=torch.long)
	all_p_mask = torch.tensor([features[t].p_mask for t in batch_index], dtype=torch.float)
	all_is_impossible = torch.tensor([features[t].is_impossible for t in batch_index], dtype=torch.float)
	all_sentid = [features[t].sentid for t in batch_index]
	all_start_positions = torch.tensor([features[t].start_position for t in batch_index], dtype=torch.long)
	all_end_positions = torch.tensor([features[t].end_position for t in batch_index], dtype=torch.long)
	return [all_input_ids, all_attention_masks , all_token_type_ids , all_cls_index , all_p_mask , all_is_impossible , all_start_positions , all_end_positions], all_sentid

def pool_sequential_embed(roberta_embed , start , end , method):
	if method =='mean':
		sub_matrix = roberta_embed[start:end+1 , :] 
		return torch.mean(sub_matrix , axis = 0 ) 

def pool_particular_embed(roberta_embed , list_index , method):
	#roberta embed shape (seq_length , hidden_size)
	if method =='mean':
		result = [] 
		for index in list_index:
			result.append(roberta_embed[index])
		return torch.mean(torch.stack(result) ,  axis = 0) #shape (hidden size) 

def find_boudaries_single_feature(single_feature):
	#this function only applying for roberta pretrained model, this not be used for other language model 
	input_ids = single_feature.input_ids
	paragraph_len = single_feature.paragraph_len
	tokens = single_feature.tokens
	sentid = single_feature.sub2sent

	len_question = len(tokens) - paragraph_len - 4 # 4 just for roberta
	end_question = len_question
	start_passage = len_question + 3
	end_passage = len(tokens) - 2 
	#bound question
	list_id_question = input_ids[1:len(tokens)-paragraph_len-3]
	list_id_text= input_ids[ len(tokens)-paragraph_len-1  :len(tokens)-1]
	bound_question =  [1 , len_question]
	bound_passage = [start_passage , end_passage]

	#bound sents
	sent_tok = {}
	for i in range(len(sentid)):
		if sentid[i] not in sent_tok:
			sent_tok[sentid[i]] = [i]
		else:
			sent_tok[sentid[i]].append(i)

	bound_sents = [] 
	for sent_id in sent_tok:
		list_sub_position = sent_tok[sent_id]
		start = list_sub_position[0] + len(list_id_question) + 3
		end = list_sub_position[len(list_sub_position) - 1] + len(list_id_question) + 3
		bound_sents.append([start,end])

	return bound_question , bound_passage , bound_sents

def find_boundaries_multifeature(features):
	bound_question , bound_features , bound_sents = [] , [] , []
	for feature in features:
		bound_ques , bound_feature , bound_sent = find_boudaries_single_feature(feature)
		bound_question = bound_ques
		bound_features.append(bound_feature)
		bound_sents.append(bound_sent)
	return bound_question , bound_features , bound_sents

class Model(nn.Module):

	def __init__(self, pretrained_model,word_emb_dim, sent_emb_dim, n_block, n_block_reasoning , device):
		super().__init__()
		self.device = device
		self.pretrained_model = pretrained_model
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.sent_emb_dim = sent_emb_dim # this value usually is set by 64 
		self.n_block = n_block 
		self.linear_pool = nn.Linear(self.word_emb_dim , self.sent_emb_dim , bias=True)
		self.n_block_reasoning = n_block_reasoning
	
		#self.sent_encoder = SentenceEncoder(self.n_block, self.sent_emb_dim, self.device)
#		self.classifier_layer = Classifier(self.sent_emb_dim, self.word_emb_dim, self.device)

		self.bottom_up_embed = Bottom_up_embed(self.word_emb_dim, self.sent_emb_dim, n_block_reasoning = self.n_block_reasoning  , chunk_dim = self.sent_emb_dim, device = self.device)

		self.W_s_e  = nn.Linear(self.word_emb_dim , 2, bias = True )

	def pool_sent_embed(self , last_hidden_state , features,  batch_index):
		batch_feature = [features[t] for t in batch_index]
		ques_embed , list_sent_emb , batch_bound_sents = self.pool_embed(last_hidden_state , batch_feature) #init embed contain question embedding and list of sentence embedding 
		return ques_embed , list_sent_emb , batch_bound_sents

	def forward(self, features, max_number_chunk = 8):
		
		squad_features = features.squad_features
		last_hidden_state_passages = self.forward_pretrained(squad_features) #shape (bs , passage_len , hidden size ) 

		bound_question , bound_passages , bound_sents = find_boundaries_multifeature(squad_features)
		
		sent2subword , subword2sents = features.sent2subword , features.subword2sents

		context_emb = self.bottom_up_embed(last_hidden_state_passages , bound_passages , sent2subword  , subword2sents) # (N word in doc + 1, word dim)

		start_end_logit = self.W_s_e(context_emb) #shape ( N-word + 1  , 2)
		start_logit = start_end_logit[: ,0 ]   # shape ( N_word + 1 )
		end_logit = start_end_logit[: ,1] # shape ( N word + 1)

		return start_logit , end_logit  

	def pad_sent(self , list_sent_emb):
		#input: list sent emb: list of tensor, each tensor has shape (N_sent in this passage , sent dim )
		#output 1: padded tensor matrix shape (bs , max sent in batch ,  sent dim )
		#output 2: length passage 
		length_passage = [t.shape[0] for t in list_sent_emb] #one type of list, which is used for storaging number of sentence for each passage
		max_sent_in_batch = max([t.shape[0] for t in list_sent_emb])
		sents_emb  = [ torch.cat( (tensor , torch.zeros(max_sent_in_batch - tensor.shape[0], tensor.shape[1]).to(torch.device('cuda')) ) , dim = 0 )  for tensor in list_sent_emb]  # list of tensor, every tensors have same shape as (max_sent_in_batch, sent_dim )
		sents_emb = torch.stack(sents_emb , dim = 0) #shape (bs , max_sent_in_batch , sent_dim ) 
		return sents_emb, length_passage

	def get_spare_embed(self , sent_hidden_state , last_hidden_state , batch_bound_sents):
		#this function is used to remove padding vector from tensor. 
		#sent hidden state: (bs , max sent , sent dim ) now will be converted into form of list of tensor (N_sent , sent dim). In this, N_sent can be different between elements 
		#last hidden state: (bs , seq length , word dim) must be converted to form: list of list of element. len of first list is batch size 
		# Second list is list of sentences in each passages. Each list can have different length 
		# Each element in list is tensor. This tensor has shape (N_words , word dim ). N_words is number of word appeared in particular sentence
		# To create the aforementioned object, we use batch bound sent  to extract number of words in sentence as well as number of sentences in passage 
		#batch bound sent is list , each element in list is 2-element list. It express start token and end token in certain sentence
		sent_result , word_result = [] , [] 
		for i in range(len(batch_bound_sents)):
			sent_result.append(sent_hidden_state[i][:len(batch_bound_sents[i])]) #only acquire real sentence 
			list_word_embed = [] 
			for j in range(len(batch_bound_sents[i])):
				start , end = batch_bound_sents[i][j][0] , batch_bound_sents[i][j][1]
				list_word_embed.append(last_hidden_state[i , start : end + 1 , : ]) #shape (end - start , word dim )
			#len list word embed = number of sentence in each passage 
			word_result.append(list_word_embed) # len word result = number of passage in batch = batch size 
		return sent_result , word_result

	def recover_original_sequence(self , last_hidden_state , paragraph_embed_start, paragraph_embed_end , batch_bound_sents):
		start_logit , end_logit = [] , [] 
		for i in range(len(paragraph_embed_start)):
			start_passage , end_passage = batch_bound_sents[i][0][0] , batch_bound_sents[i][-1][1]
			start_logit.append(torch.cat( (last_hidden_state[i , : start_passage , : ] , paragraph_embed_start[i], last_hidden_state[i , end_passage + 1 : , :] ) ))
			end_logit.append(torch.cat( (last_hidden_state[i , : start_passage , : ] , paragraph_embed_end[i], last_hidden_state[i , end_passage + 1  :, :] ) ))
		return torch.stack(start_logit , dim  = 0) , torch.stack(end_logit , dim = 0) #shape (bs , seq length , word dim )

	def forward_pretrained(self, features , max_batch = 8 ):
		all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
		output = []
		for i in range(all_input_ids.shape[0]):
			input_id= all_input_ids[i, : ].unsqueeze(0)
			t = self.pretrained_model(input_id)[0] #shape (1 , N_word , word dim)
			output.append(t.squeeze(0)) #shape (N_word , word dim )
		output = torch.stack(output ) #shape (N_chunk , N_word , word dim)
		return output 

	def pool_embed(self, roberta_embed, features , method_pool = 'mean'):
	#roberta embed of each sample in a batch: shape (sequen_length , hidden size)
		results = [] #results is list, one element in result correspond with triple value (question embed , sents_embed, unique_subword_embed) of single feature
		#features can be understood as batch of features
		batch_bound_sents = []
		ques_embed , sents_embed = [] , []
		for i, feature in enumerate(features):
			bound_question, bound_sents  = find_boudaries_single_feature(feature)
			batch_bound_sents.append(bound_sents)
			
			question_embed = pool_sequential_embed(roberta_embed[i] , bound_question[0] , bound_question[1] , method_pool) #shape (hidden_size)
			question_embed = torch.tanh(self.linear_pool(question_embed)) #shape (bs , sent dim )
			#sent emb has different dim with word emb
			batch_sent_embed = [] 
			for bound_sent in bound_sents:
				batch_sent_embed.append(torch.tanh(self.linear_pool(pool_sequential_embed(roberta_embed[i] , bound_sent[0] , bound_sent[1] , method_pool) )))
			batch_sent_embed = torch.stack(batch_sent_embed , axis = 0) #shape (num_sent , hidden_size)	
			
			sents_embed.append(batch_sent_embed)
			ques_embed.append(question_embed)
		ques_embed = torch.stack(ques_embed) #shape (bs , sent dim )
		return ques_embed , sents_embed , batch_bound_sents
