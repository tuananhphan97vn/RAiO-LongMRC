
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

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)
# print(FILTERWORD)

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

def _new_check_is_max_context(doc_spans, cur_span_index, position):
	"""Check if this is the 'max context' doc span for the token."""
	# if len(doc_spans) == 1:
	# return True
	best_score = None
	best_span_index = None
	for (span_index, doc_span) in enumerate(doc_spans):
		end = doc_span["start"] + doc_span["length"] - 1
		if position < doc_span["start"]:
			continue
		if position > end:
			continue
		num_left_context = position - doc_span["start"]
		num_right_context = end - position
		score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
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

	#need one object to matching sents vs corresponding words
	features = []
	if is_training and not example.is_impossible:
		# Get start and end position
		start_position = example.start_position
		end_position = example.end_position

		# If the answer cannot be found in the text, then skip this example.
		actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])  #answer text 
		cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
		if actual_text.find(cleaned_answer_text) == -1:
			logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
			return []

	tok_to_orig_index = []
	orig_to_tok_index = []
	all_doc_tokens = []

	word_to_sent_index = []
	# sent_to_word_index = [] 
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
			"LongformerTokenizer",
			"BartTokenizer",
			"RobertaTokenizerFast",
			"LongformerTokenizerFast",
			"BartTokenizerFast",
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

	#create sent2subword
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

	# Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
	# in the way they compute mask of added tokens.
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
		# print('sub2sent ' , sub2sent)

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
		# cls_index = span["input_ids"].index(tokenizer.cls_token_id)
		# p_mask = np.ones_like(span["token_type_ids"])
		# if tokenizer.padding_side == "right":
		# 	p_mask[len(truncated_query) + sequence_added_tokens :] = 0
		# else:
		# 	p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

		# pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
		# special_token_indices = np.asarray(
		# 	tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
		# ).nonzero()

		# p_mask[pad_token_indices] = 1
		# p_mask[special_token_indices] = 1

		# p_mask[cls_index] = 0

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
	"""
	Converts a list of examples into a list of features that can be directly given as input to a model. It is
	model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

	Args:
		examples: list of :class:`~transformers.data.processors.squad.SquadExample`
		tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
		max_seq_length: The maximum sequence length of the inputs.
		doc_stride: The stride used when the context is too large and is split across several features.
		max_query_length: The maximum length of the query.
		is_training: whether to create features for model evaluation or model training.
		padding_strategy: Default to "max_length". Which padding strategy to use
		return_dataset: Default False. Either 'pt' or 'tf'.
			if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
		threads: multiple processing threads.

	Returns:
		list of :class:`~transformers.data.processors.squad.SquadFeatures`

	Example::

		processor = SquadV2Processor()
		examples = processor.get_dev_examples(data_dir)

		features = squad_convert_examples_to_features(
			examples=examples,
			tokenizer=tokenizer,
			max_seq_length=args.max_seq_length,
			doc_stride=args.doc_stride,
			max_query_length=args.max_query_length,
			is_training=not evaluate,
		)
	"""
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
	# if return_dataset == "pt":
	# 	if not is_torch_available():
	# 		raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

	# 	# Convert to Tensors and build dataset
	# 	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	# 	all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
	# 	all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
	# 	all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
	# 	all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
	# 	all_is_impossible = torch.tensor([f.is_impossible for f in features])
	# 	all_sentid = [f.sentid for f in features]


	if not is_training:
		# all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
		# dataset = TensorDataset(
		# 	all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
		# )
		return features , examples
	else:
		# all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
		# all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
		# dataset = TensorDataset(
		# 	all_input_ids,
		# 	all_attention_masks,
		# 	all_token_type_ids,
		# 	all_start_positions,
		# 	all_end_positions,
		# 	all_cls_index,
		# 	all_p_mask,
		# 	all_is_impossible,
		# )

# 	return features, dataset , all_sentid
# else:
# 	return features
		return features


