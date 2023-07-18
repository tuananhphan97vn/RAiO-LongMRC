from nltk.tokenize import word_tokenize
from handle_text_bert import sent_tokenize
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from tqdm import tqdm
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from transformers.utils import logging
from transformers.data.processors.utils import DataProcessor
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}
FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)
if is_torch_available():
	import torch
	from torch.utils.data import TensorDataset
if is_tf_available():
	import tensorflow as tf
logger = logging.get_logger(__name__)
def filter_sent(sent, filter_words):
	sent = sent.lower()
	return [ word for word in word_tokenize(sent) if word not in filter_words]
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
	"""Returns tokenized answer spans that better match the annotated answer."""
	tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

	for new_start in range(input_start, input_end + 1):
		for new_end in range(input_end, new_start - 1, -1):
			text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
			if text_span == tok_answer_text:
				return (new_start, new_end)
	return (input_start, input_end)
def _check_is_max_context(doc_spans, cur_span_index, position):
	"""Check if this is the 'max context' doc span for the token."""
	best_score = None
	best_span_index = None
	for (span_index, doc_span) in enumerate(doc_spans):
		end = doc_span.start + doc_span.length - 1
		if position < doc_span.start:
			continue
		if position > end:
			continue
		num_left_context = position - doc_span.start
		num_right_context = end - position
		score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
		if best_score is None or score > best_score:
			best_score = score
			best_span_index = span_index
	return cur_span_index == best_span_index
def _is_whitespace(c):
	if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
		return True
	return False
def squad_convert_example_to_features(
	example, max_seq_length, max_len_context, max_query_length, padding_strategy, is_training
):
	features = []
	if is_training and not example.is_impossible:
		# Get start and end position
		start_position = example.start_position
		end_position = example.end_position
		actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])  #answer text 
		cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
		if actual_text.find(cleaned_answer_text) == -1:
			logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
			return []
	tok_to_orig_index = []
	orig_to_tok_index = []
	all_doc_tokens = []
	word_to_sent_index = []
	original_doc = " ".join(example.doc_tokens)
	sents = sent_tokenize(original_doc)
	if sum([len(sent.split()) for sent in sents]) != len(example.doc_tokens):
		print('erorr tokenize sent ')
		return 
	current_num_word = 0 
	for i , sent in enumerate(sents):
		words = sent.split()
		for word in words :
			word_to_sent_index.append(i)
	subword2sent = []
	for (i, token) in enumerate(example.doc_tokens):
		orig_to_tok_index.append(len(all_doc_tokens))
		if tokenizer.__class__.__name__ in [
			"RobertaTokenizer",
		]:
			sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
			for sub_token in sub_tokens:
				subword2sent.append(word_to_sent_index[i])
		else:
			sub_tokens = tokenizer.tokenize(token)
			for sub_token in sub_tokens:
				subword2sent.append(word_to_sent_index[i])
		for sub_token in sub_tokens:
			tok_to_orig_index.append(i)
			all_doc_tokens.append(sub_token)
	if  is_training and not example.is_impossible:
		tok_start_position = orig_to_tok_index[example.start_position]
		if example.end_position < len(example.doc_tokens) - 1:
			tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
		else:
			tok_end_position = len(all_doc_tokens) - 1

		(tok_start_position, tok_end_position) = _improve_answer_span(
			all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
		)
	sent2subword = {}
	for i in range(len(subword2sent)):
		if subword2sent[i] not in sent2subword:
			sent2subword[subword2sent[i]] = [i]
		else:
			sent2subword[subword2sent[i]].append(i)
	spans = []
	truncated_query = tokenizer.encode(
		example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
	)
	tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
	sequence_added_tokens = (
		tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
		if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
		else tokenizer.model_max_length - tokenizer.max_len_single_sentence
	)
	sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
	span_doc_tokens = all_doc_tokens
	len_doc = len(all_doc_tokens)
	#split doc to multiple chunk with max len context 
	contexts = [ all_doc_tokens[i:i+max_len_context] for i in range(0 , len_doc , max_len_context)]
	subword2sents = [subword2sent[i:i+max_len_context] for i in range(0 , len_doc, max_len_context)]
	for i in range(len(contexts)):
		context = contexts[i] #context of this chunk. We split document to multiple chunks due to limitation of BERT
		sub2sent = subword2sents[i] #subword 2 sent only in one chunk 
		if tokenizer.padding_side == "right": 
			texts = truncated_query
			pairs = context 
			truncation = TruncationStrategy.ONLY_SECOND.value
		encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
			texts,
			pairs,
			truncation=truncation,
			padding=padding_strategy,
			max_length=max_seq_length,
			return_token_type_ids=True,
		)
		if tokenizer.pad_token_id in encoded_dict["input_ids"]:
			if tokenizer.padding_side == "right":
				non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
		else:
			non_padded_ids = encoded_dict["input_ids"]

		tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
		encoded_dict["paragraph_len"] = len(context)
		encoded_dict["tokens"] = tokens
		encoded_dict['sub2sent'] = sub2sent
		spans.append(encoded_dict)
	for span in spans:
		features.append(
			SquadProposeFeatures(
				span["input_ids"],
				span['paragraph_len'],
				span['sub2sent'],
				example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
				unique_id=0,
				tokens=span["tokens"],
				qas_id=example.qas_id,
			)
		)
	start_position = 0
	end_position = 0
	if is_training and not example.is_impossible:
		start_position = tok_start_position + 1 
		end_position = tok_end_position + 1 
	propose_example = ProposeExample(features , start_position , end_position , all_doc_tokens , tok_to_orig_index , sent2subword , subword2sents)
	return propose_example
def squad_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
	global tokenizer
	tokenizer = tokenizer_for_convert
def squad_convert_examples_to_features(
	examples,
	tokenizer,
	max_seq_length,
	max_len_context,
	max_query_length,
	is_training,
	padding_strategy="max_length",
	return_dataset=False,
	threads=1,
	tqdm_enabled=True,
):
	# Defining helper methods
	features = []
	threads = min(threads, cpu_count())
	with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
		annotate_ = partial(
			squad_convert_example_to_features,
			max_seq_length=max_seq_length,
			max_len_context=max_len_context,
			max_query_length=max_query_length,
			padding_strategy=padding_strategy,
			is_training=is_training,
		)
		features = list(
			tqdm(
				p.imap(annotate_, examples, chunksize=32),
				total=len(examples),
				desc="convert squad examples to features",
				disable=not tqdm_enabled,
			)
		)
	new_features = []
	unique_id = 1000000000
	example_index = 0
	for example_feature in tqdm(
		features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
	):
		if not example_feature:
			continue
		# for example_feature in example_features:
		example_feature.example_index = example_index
		example_feature.unique_id = unique_id
		new_features.append(example_feature)
		unique_id += 1
		example_index += 1
	features = new_features
	del new_features
	if not is_training:
		return features , examples
	else:
		return features
class SquadProcessor(DataProcessor):
	train_file = None
	dev_file = None
	def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
		if not evaluate:
			answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
			answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
			answers = []
		else:
			answers = [
				{"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
				for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
			]
			answer = None
			answer_start = None
		return SquadExample(
			qas_id=tensor_dict["id"].numpy().decode("utf-8"),
			question_text=tensor_dict["question"].numpy().decode("utf-8"),
			context_text=tensor_dict["context"].numpy().decode("utf-8"),
			answer_text=answer,
			start_position_character=answer_start,
			title=tensor_dict["title"].numpy().decode("utf-8"),
			answers=answers,
		)
	def get_examples_from_dataset(self, dataset, evaluate=False):
		if evaluate:
			dataset = dataset["validation"]
		else:
			dataset = dataset["train"]
		examples = []
		for tensor_dict in tqdm(dataset):
			examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))
		return examples
	def get_train_examples(self, data_dir, filename=None):
		if data_dir is None:
			data_dir = ""
		if self.train_file is None:
			raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")
		with open(
			os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
		) as reader:
			input_data = json.load(reader)["data"]
		return self._create_examples(input_data, "train")
	def get_dev_examples(self, data_dir, filename=None):
		if data_dir is None:
			data_dir = ""
		if self.dev_file is None:
			raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")
		with open(
			os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
		) as reader:
			input_data = json.load(reader)["data"]
		return self._create_examples(input_data, "dev")
	def _create_examples(self, input_data, set_type):
		is_training = set_type == "train"
		examples = []
		for entry in tqdm(input_data):
			# title = entry["title"]
			for paragraph in entry["paragraphs"]:
				context_text = paragraph["context"]
				for qa in paragraph["qas"]:
					qas_id = qa["id"]
					question_text = qa["question"]
					start_position_character = None
					answer_text = None
					answers = []
					is_impossible = qa.get("is_impossible", False)
					if not is_impossible:
						if is_training:
							answer = qa["answers"][0]
							answer_text = answer["text"]
							start_position_character = answer["answer_start"]
						else:
							answers = qa["answers"]

					example = SquadExample(
						qas_id=qas_id,
						question_text=question_text,
						context_text=context_text,
						answer_text=answer_text,
						start_position_character=start_position_character,
						title="empty",
						is_impossible=is_impossible,
						answers=answers,
					)
					examples.append(example)
		return examples
class SquadExample:
	def __init__(
		self,
		qas_id,
		question_text,
		context_text,
		answer_text,
		start_position_character,
		title,
		answers=[],
		is_impossible=False,
	):
		self.qas_id = qas_id
		self.question_text = question_text
		self.context_text = context_text
		self.answer_text = answer_text
		self.title = title
		self.is_impossible = is_impossible
		self.answers = answers
		self.start_position, self.end_position = 0, 0
		doc_tokens = []
		char_to_word_offset = []
		prev_is_whitespace = True
		for c in self.context_text:
			if _is_whitespace(c):
				prev_is_whitespace = True
			else:
				if prev_is_whitespace:
					doc_tokens.append(c)
				else:
					doc_tokens[-1] += c
				prev_is_whitespace = False
			char_to_word_offset.append(len(doc_tokens) - 1)
		self.doc_tokens = doc_tokens 
		self.char_to_word_offset = char_to_word_offset
		if start_position_character is not None and not is_impossible:
			self.start_position = char_to_word_offset[start_position_character]
			self.end_position = char_to_word_offset[
				min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
			]
class SquadProposeFeatures:
	def __init__(
		self,
		input_ids,
		paragraph_len,
		sub2sent,
		example_index,
		unique_id,
		tokens,
		qas_id: str = None,
		encoding: BatchEncoding = None,
	):
		self.input_ids = input_ids
		self.paragraph_len = paragraph_len
		self.sub2sent = sub2sent # list 
		self.example_index = example_index
		self.unique_id = unique_id
		self.tokens = tokens
		self.qas_id = qas_id
		self.encoding = encoding
class ProposeExample:
	def __init__(
		self,
		squad_features, start_position, end_position, full_tokens , tok_to_orig_index , sent2subword , subword2sents):
		self.squad_features = squad_features
		self.start_position = start_position
		self.end_position = end_position
		self.tokens = full_tokens
		self.tok_to_orig_index = tok_to_orig_index
		self.sent2subword = sent2subword
		self.subword2sents = subword2sents
class SquadResult:
	def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
		self.start_logits = start_logits
		self.end_logits = end_logits
		self.unique_id = unique_id
