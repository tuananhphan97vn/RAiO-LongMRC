class Sent2WordBlock(nn.Module):

	def __init__(self , sent_dim):
		super().__init__()
		self.sent_dim = sent_dim

		self.sent2word_multiblock = MultiBlockSelfAttention(self.sent_dim , 2)

	def forward(self,  list_sent_hid , list_word_hidden , bound_passages, subword2sents):
		ques_length = bound_passages[0][0] # ques_len + number special tokens
		result = [] 
		for i in range(len(list_word_hidden)):
			ques_hidden , word_hiddden = list_word_hidden[i][:ques_length , :] , list_word_hidden[i][ques_length: , :]
			#compute sent hidden state before/end current chunk 
			if i == 0: 
				# back sent = 0 
				original_len = list_word_hidden[i].shape[0]

				next_sents = [list_sent_hid[j] for j in range(len(list_word_hidden)) if j > 0 ]
				if len(next_sents) > 0 :
					next_sents = torch.cat(next_sents)
					ques_word_sent = torch.cat([ques_hidden , word_hiddden , next_sents])
				else:
					ques_word_sent = list_word_hidden[i]
				_ = self.sent2word_multiblock(ques_word_sent)

				result.append(_[ : original_len , : ])

			if 0 < i and i < len(list_word_hidden) - 1 :
				back_sents , next_sents = [] , []
				back_sents = [list_sent_hid[j] for j in range(len(list_word_hidden)) if j < i]
				next_sents = [list_sent_hid[j] for j in range(len(list_word_hidden)) if j > i]
				back_sents = torch.cat(back_sents)
				next_sents = torch.cat(next_sents)

				num_back_sents = back_sents.shape[0]
				num_next_sents = next_sents.shape[0]

				ques_word_sent = torch.cat([ques_hidden , back_sents , word_hiddden , next_sents]) 
				_ = self.sent2word_multiblock(ques_word_sent)
				original_len  = list_word_hidden[i].shape[0]
				new_hid = torch.cat( [_[:ques_length , : ] , _[ques_length + num_back_sents : list_word_hidden[i].shape[0] + num_back_sents, : ] ])

				if new_hid.shape[0] != list_word_hidden[i].shape[0]:
					print('not match shape [0]')
				result.append(new_hid)

			if i == len(list_word_hidden) - 1 and i > 0 :
				back_sents = [list_sent_hid[j] for j in range(len(list_word_hidden)) if j < len(list_word_hidden) - 1 ]
				back_sents = torch.cat(back_sents)
				num_back_sents = back_sents.shape[0]

				ques_word_sent = torch.cat([ques_hidden , back_sents , word_hiddden])
				_ = self.sent2word_multiblock(ques_word_sent)

				new_hid = torch.cat( [_[:ques_length , : ] , _[ques_length + num_back_sents : list_word_hidden[i].shape[0] + num_back_sents ] ] )
				
				if new_hid.shape[0] != list_word_hidden[i].shape[0]:
					print('not match shape 0 ')
				result.append(new_hid)
		return result
class QuestionSentenceWordReasoning(nn.Module):

	def __init__(self , sent_dim ,device ) -> None:

		super().__init__()
		self.sent_dim = sent_dim
		self.device = device
		self.sent2word = Sent2WordBlock(self.sent_dim)
		self.word2sent = Word2SentBlock(self.sent_dim , self.device)
		self.sent2sent = Sent2SentBlock(self.sent_dim)

	def forward(self,  list_word_hid , bound_passages , subword2sents):
		list_sent_hid = self.word2sent(list_word_hid, bound_passages, subword2sents) # list of tensor , each tensor shaoe (num sent in chunk , sent dim ) 
		list_sent_hid = self.sent2sent(list_sent_hid)
		list_word_hid= self.sent2word(list_sent_hid , list_word_hid , bound_passages , subword2sents)
		return list_word_hid
