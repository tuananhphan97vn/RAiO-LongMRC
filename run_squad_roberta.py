
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

from audioop import cross
import pickle

import argparse
import glob
import logging
from typing import overload
logger = logging.getLogger(__name__)
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from model import Model
import transformers
from transformers import (
	MODEL_FOR_QUESTION_ANSWERING_MAPPING,
	WEIGHTS_NAME,
	AdamW,
	AutoConfig,
	AutoModelForQuestionAnswering,
	AutoTokenizer,
	get_linear_schedule_with_warmup,
	squad_convert_examples_to_features,
)
from squad_metrics import (
	compute_predictions_logits,
	squad_evaluate,
)
from modify_squad import *
from transformers.trainer_utils import is_main_process
import torch.nn as nn

# nll_loss = nn.NLLLoss(reduction='mean')
crossE_loss= nn.CrossEntropyLoss(reduction='mean')
bce_loss =  nn.BCEWithLogitsLoss(reduction='mean')

try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter


MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
	return tensor.detach().cpu().tolist()

def iniGraphEmbedding(outputs):
	return sentEmbed, uniqueSubWordEmbed

def generate_graph_embedding(origins, graphWord2Word , graphWord2Sent,  graphSent2Sent):
	hidden_states = outputs
	return hidden_states

def combineRobertaVsGraph(origins, graph_embed):
	return hidden_states


# [list_triple_graph , list_map] = pickle.load(open('triple_graph.sav','rb'))
import time 
def train(args, features, model, tokenizer):

	"""Train the model"""
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) 

	num_train_data = len(features)
	t_total = num_train_data // args.gradient_accumulation_steps * args.num_train_epochs

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters() ), lr=args.learning_rate)

	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	# Check if saved optimizer or scheduler states exist
	if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
		os.path.join(args.model_name_or_path, "scheduler.pt")
	):
		# Load in optimizer and scheduler states
		optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
		scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
		)
	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(features))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. parallel, distributed & accumulation) = %d",
		args.train_batch_size
		* args.gradient_accumulation_steps
		* (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
	)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 1
	epochs_trained = 0
	steps_trained_in_current_epoch = 0
	# Check if continuing training from a checkpoint
	if os.path.exists(args.model_name_or_path):
		try:
			# set global_step to gobal_step of last saved checkpoint from model path
			checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
			global_step = int(checkpoint_suffix)
			epochs_trained = global_step // ( num_train_data // args.gradient_accumulation_steps)
			steps_trained_in_current_epoch = global_step % ( num_train_data // args.gradient_accumulation_steps)
			logger.info("  Continuing training from checkpoint, will skip to saved global_step")
			logger.info("  Continuing training from epoch %d", epochs_trained)
			logger.info("  Continuing training from global step %d", global_step)
			logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

		except ValueError:
			logger.info("  Starting fine-tuning.")

	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(
		epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
	)
	# Added here for reproductibility
	set_seed(args)

	best_F1 = []
	best_EM = [] 
	best_result = 0.0 
	N_down_lr , count =  3,  0 
	min_lr = 1e-7  # The value of learning rate that model can be down to, if learning rate reach this value, it is not decreased more
	#module is desinged to allow .. reduce learning rate in training process 
	decay_rate = 0.2
	original_lr = args.learning_rate
	decay_lr_epoch = original_lr / int(args.num_train_epochs)
	print('first evaluation ...')
	#results = evaluate(args, model, tokenizer)

	for epoch in range(int(args.num_train_epochs)): 

		#for each epoch, recompute learing rate 
		args.learning_rate = original_lr - decay_lr_epoch * epoch 
		overall_train_loss = 0 
		print('new learing rate ' , args.learning_rate ) 
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters() ), lr=args.learning_rate ) 

		original_index = list(range(len(features)))
		np.random.shuffle(original_index)
		list_index_batch = [original_index[x:x+args.train_batch_size] for x in range(0, len(original_index), args.train_batch_size)]

		for step,  batch_index  in enumerate(list_index_batch):
			global_step += 1

			# Skip past any already trained steps if resuming training
			if steps_trained_in_current_epoch > 0 :
				steps_trained_in_current_epoch -= 1
				continue
			
			model.train()	
			# try:	
			squad_features = features[batch_index[0]]
			start_position = torch.tensor(squad_features.start_position , dtype=torch.long).to(args.device)
			end_position = torch.tensor(squad_features.end_position , dtype=torch.long).to(args.device)

			word_logit_start, word_logit_end = model.forward( squad_features, batch_index) #shape (N word in doc + 1)

			# print(word_logit_end.shape , end_position)
			loss_start = crossE_loss(word_logit_start.to(args.device), start_position) 
			loss_end  = crossE_loss(word_logit_end.to(args.device) , end_position)			

			loss = loss_start + loss_end  
			
			overall_train_loss += loss 
			if step % 2000 == 0 :
				print('loss ' , loss)
			if step % args.logging_steps == 0 :
				print('learning rate ' , args.learning_rate , 'epoch ', epoch , ' step ', step , 'loss', loss )
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps
			loss.backward()
			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				if args.fp16:
					torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
				else:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				# print('update .')
				optimizer.step()
				scheduler.step()  
				model.zero_grad()
			# 	# Log metrics 
			if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
				# Only evaluate when single GPU otherwise metrics may not average well
				if args.local_rank == -1 and args.evaluate_during_training:
					results = evaluate(args, model, tokenizer)

					current_f1 = results['f1']
					current_EM = results['exact']


					if len(best_F1) < 3 :
						best_F1.append(current_f1)
						print('three best f1 ' , best_F1)
						output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
						torch.save(model.state_dict() , output_dir)
						logger.info("Saving optimizer and scheduler states to %s", output_dir)
					else:
						curr_min_f1 = min(best_F1)
						if current_f1 > curr_min_f1: #bestF1 now is list of results

							best_F1.remove(curr_min_f1)
							best_F1.append(current_f1)
							
							print('current f1 ', current_f1 , 'three best F1 ', best_F1, 'current result ' , results )

							#save current best model 
							output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
							torch.save(model.state_dict() , output_dir)
							logger.info("Saving optimizer and scheduler states to %s", output_dir)
					
					if len(best_EM) < 3 :
						best_EM.append(current_EM)
						print('three best EM ' , best_EM)
						output_dir = os.path.join(args.output_dir, "checkpointEM-{}".format(global_step))
						torch.save(model.state_dict() , output_dir)
						logger.info("Saving optimizer and scheduler states to %s", output_dir)							
					else:
						curr_min_em = min(best_EM)
						if current_EM > curr_min_em:

							best_EM.remove(curr_min_em)
							best_EM.append(current_EM)

							print('current EM ', current_EM , 'three best EM', best_EM , 'current result' , results )

							#save current best model 
							output_dir = os.path.join(args.output_dir, "checkpointEM-{}".format(global_step))
							torch.save(model.state_dict() , output_dir)
							logger.info("Saving optimizer and scheduler states to %s", output_dir)							

					# else:
					# 	#In case current F1 is not greater than best F1 
					# 	count += 1 
					# 	print('best f1 ', best_F1 , 'all result ', best_result )

					for key, value in results.items():
						tb_writer.add_scalar("eval_{}".format(key), value, global_step)

				tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
				tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
				logging_loss = tr_loss

			# except Exception as e :
			# 	print('exception is raised  ' , e)
				#print('error matching size between target and input with target size ' , all_is_impossible.size() , 'and input size ', sent_digit.size())

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break

		print('overall train loss ', overall_train_loss)
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

	if args.local_rank in [-1, 0]:
		tb_writer.close()

	return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
	print('evaluating ..... ')
	features, examples= load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
	#dataset, examples, features = dataset[:100], examples[:100], features[:100]
	# all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	# all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

	# list_sent_id = list_sent_id
	if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(args.output_dir)

	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

	# multi-gpu evaluate
	if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
		model = torch.nn.DataParallel(model)

	# Eval!
	logger.info("***** Running evaluation {} *****".format(prefix))
	logger.info("  Num examples = %d", len(features))
	logger.info("  Batch size = %d", args.eval_batch_size)

	all_results = []
	start_time = timeit.default_timer()

	#this index depend on index of batch of graph  
	original_index = list(range(len(features)))
	list_index_batch = [original_index[x:x+args.eval_batch_size] for x in range(0, len(original_index), args.eval_batch_size)]

	print('starting evaluation .... ')
	overall_loss = 0
	with torch.no_grad():
		for step,  batch_index  in enumerate(list_index_batch):
		
			model.eval()	

			squad_features = features[batch_index[0]]

			word_logit_start, word_logit_end = model.forward( squad_features, batch_index) #shape (number token in document + 1 )
			
			start_position = torch.tensor(squad_features.start_position , dtype=torch.long).to(args.device)
			end_position = torch.tensor(squad_features.end_position , dtype=torch.long).to(args.device)

			loss_start = crossE_loss(word_logit_start.to(args.device), start_position) 
			loss_end  = crossE_loss(word_logit_end.to(args.device) , end_position) 

			loss = (loss_start + loss_end ) / 2 

			overall_loss += loss 

			feature_index = batch_index[0]

			eval_feature = features[feature_index]
			unique_id = int(eval_feature.unique_id)

			start_logit, end_logit = to_list(word_logit_start) , to_list(word_logit_end) 
			# print('start logits ', type(start_logits))
			result = SquadResult(unique_id, start_logit, end_logit)

			all_results.append(result)

	evalTime = timeit.default_timer() - start_time
	print("  Evaluation done in total %f secs (%f sec per example) , loss ", evalTime, evalTime / len(features) , overall_loss)

	# Compute predictions
	output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
	output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

	if args.version_2_with_negative:
		output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
	else:
		output_null_log_odds_file = None

	# XLNet and XLM use a more complex post-processing procedure

	predictions = compute_predictions_logits(
		examples,
		features,
		all_results,
		args.n_best_size,
		args.max_answer_length,
		args.do_lower_case,
		output_prediction_file,
		output_nbest_file,
		output_null_log_odds_file,
		args.verbose_logging,
		args.version_2_with_negative,
		args.null_score_diff_threshold,
		tokenizer,
	)

	# Compute the F1 and exact scores.
	results = squad_evaluate(examples, predictions)
	print("Results: {}".format(results))
	return results

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):

	print('evaluate status ' , evaluate , output_examples)

	if args.local_rank not in [-1, 0] and not evaluate:
		# Make sure only the first process in distributed training process the dataset, and the others will use the cache
		torch.distributed.barrier()

	# Load data features from cache or dataset file
	input_dir = args.data_dir if args.data_dir else "."
	cached_features_file = os.path.join(
		input_dir,
		"cached_{}_{}_{}".format(
			"dev" if evaluate else "train",
			list(filter(None, args.model_name_or_path.split("/"))).pop(),
			str(args.max_seq_length),
		),
	)

	# Init features and dataset from cache if it exists
	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		import pickle
		logger.info("Loading features from cached file %s", cached_features_file)
		# features_and_dataset = torch.load(cached_features_file)
		file = open(cached_features_file , 'rb')
		features_and_dataset = pickle.load(file)

		features = features_and_dataset['features']
	else:
		logger.info("Creating features from dataset file at %s", input_dir)

		if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
			try:
				import tensorflow_datasets as tfds
			except ImportError:
				raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

			if args.version_2_with_negative:
				logger.warning("tensorflow_datasets does not handle version 2 of SQuAD.")

			tfds_examples = tfds.load("squad")
			examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
		else:
			processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
			if evaluate:
				examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
			else:
				examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

		features = squad_convert_examples_to_features(
			examples=examples,
			tokenizer=tokenizer,
			max_seq_length=args.max_seq_length,
			max_len_context=args.max_len_context,
			max_query_length=args.max_query_length,
			# chunk_length=args.chunk_length,
			is_training=not evaluate,
			return_dataset="pt",
			threads=args.threads,
		)

		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file %s", cached_features_file)
			# torch.save({"features": features, "dataset": dataset, "examples": examples, 'list_sent_id':list_sent_id}, cached_features_file)
			# torch.save({"features": features}, cached_features_file)
			import pickle 
			file = open(cached_features_file , 'wb')
			pickle.dump({"features": features}, file)
			print('finish writing file .....')

	if args.local_rank == 0 and not evaluate:
		# Make sure only the first process in distributed training process the dataset, and the others will use the cache
		torch.distributed.barrier()

	if output_examples:
		# return dataset, examples, features , list_sent_id
		return features 

	# return dataset , list_sent_id
	# return features, dataset  , list_sent_id
	return features 

def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--model_type",
		default=None,
		type=str,
		required=True,
		help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
	)
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		required=True,
		help="Path to pretrained model or model identifier from huggingface.co/models",
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		required=True,
		help="The output directory where the model checkpoints and predictions will be written.",
	)

	# Other parameters
	parser.add_argument(
		"--data_dir",
		default=None,
		type=str,
		help="The input data dir. Should contain the .json files for the task."
		+ "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
	)
	parser.add_argument(
		"--train_file",
		default=None,
		type=str,
		help="The input training file. If a data dir is specified, will look for the file there"
		+ "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
	)
	parser.add_argument(
		"--predict_file",
		default=None,
		type=str,
		help="The input evaluation file. If a data dir is specified, will look for the file there"
		+ "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
	)
	parser.add_argument(
		"--config_name", default="", type=str,
		help="Pretrained config name or path if not the same as model_name"
	)
	parser.add_argument(
		"--tokenizer_name",
		default="",
		type=str,
		help="Pretrained tokenizer name or path if not the same as model_name",
	)
	parser.add_argument(
		"--cache_dir",
		default="",
		type=str,
		help="Where do you want to store the pre-trained models downloaded from huggingface.co",
	)

	parser.add_argument(
		"--version_2_with_negative",
		action="store_true",
		help="If true, the SQuAD examples contain some that do not have an answer.",
	)
	parser.add_argument(
		"--null_score_diff_threshold",
		type=float,
		default=0.0,
		help="If null_score - best_non_null is greater than the threshold predict null.",
	)

	parser.add_argument(
		"--max_seq_length",
		default=384,
		type=int,
		help="The maximum total input sequence length after WordPiece tokenization. Sequences "
		"longer than this will be truncated, and sequences shorter than this will be padded.",
	)
	parser.add_argument(
		"--max_len_context",
		default=128,
		type=int,
		help="When splitting up a long document into chunks, how much stride to take between chunks.",
	)
	parser.add_argument(
		"--max_query_length",
		default=64,
		type=int,
		help="The maximum number of tokens for the question. Questions longer than this will "
		"be truncated to this length.",
	)
	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument(
		"--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
	)
	parser.add_argument(
		"--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
	)

	parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument(
		"--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
	)
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
	)
	parser.add_argument(
		"--max_steps",
		default=-1,
		type=int,
		help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
	)
	parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
	parser.add_argument(
		"--n_best_size",
		default=20,
		type=int,
		help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
	)
	parser.add_argument(
		"--max_answer_length",
		default=30,
		type=int,
		help="The maximum length of an answer that can be generated. This is needed because the start "
		"and end predictions are not conditioned on one another.",
	)
	parser.add_argument(
		"--verbose_logging",
		action="store_true",
		help="If true, all of the warnings related to data processing will be printed. "
		"A number of warnings are expected for a normal SQuAD evaluation.",
	)
	parser.add_argument(
		"--lang_id",
		default=0,
		type=int,
		help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
	)

	parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
	parser.add_argument(
		"--eval_all_checkpoints",
		action="store_true",
		help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
	)
	parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
	parser.add_argument(
		"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
	)
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

	parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
	)
	parser.add_argument(
		"--fp16_opt_level",
		type=str,
		default="O1",
		help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
		"See details at https://nvidia.github.io/apex/amp.html",
	)
	parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
	parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

	parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
	parser.add_argument("--niter_graph", type =int , default = 1 , help="how many iterations for carrying out iteratively updater for GNN")
	parser.add_argument("--n_block", type =int , default = 1 , help="")
	parser.add_argument("--word_dim", type =int , default = 1024 , help="word dim for hidden dim of pretrained model ")
	parser.add_argument("--sent_dim", type =int , default = 64 , help="sent dim. This value can be varied to find best results")
	parser.add_argument("--chunk_length", type =int , default = 10 , help="sent dim. This value can be varied to find best results")
	parser.add_argument("--n_block_reasoning", type =int , default = 2 , help="number of block chunk-word cross information")
	args = parser.parse_args()



	if (
		os.path.exists(args.output_dir)
		and os.listdir(args.output_dir)
		and args.do_train
		and not args.overwrite_output_dir
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir
			)
		)

	# Setup distant debugging if needed
	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd

		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend="nccl")
		args.n_gpu = 1
	args.device = device

	if is_main_process(args.local_rank):
		transformers.utils.logging.set_verbosity_info()
		transformers.utils.logging.enable_default_handler()
		transformers.utils.logging.enable_explicit_format()
	# Set seed
	set_seed(args)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		# Make sure only the first process in distributed training will download model & vocab
		torch.distributed.barrier()

	args.model_type = args.model_type.lower()
	config = AutoConfig.from_pretrained(
		args.config_name if args.config_name else args.model_name_or_path,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)

	autoQAmodel = AutoModelForQuestionAnswering.from_pretrained(
		args.model_name_or_path,
		from_tf=bool(".ckpt" in args.model_name_or_path),
		config=config,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)
	tokenizer = AutoTokenizer.from_pretrained(
		args.model_name_or_path,
		do_lower_case=args.do_lower_case,
		cache_dir=args.cache_dir if args.cache_dir else None,
		use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
	)
	model_roberta  = autoQAmodel.roberta

	if args.local_rank == 0:
		# Make sure only the first process in distributed training will download model & vocab
		torch.distributed.barrier()

	if args.local_rank == 0:
		# Make sure only the first process in distributed training will download model & vocab
		torch.distributed.barrier()

	model = Model(pretrained_model = model_roberta, word_emb_dim = args.word_dim, sent_emb_dim = args.sent_dim , n_block= args.n_block, n_block_reasoning = args.n_block_reasoning , device = args.device)
	# model.load_state_dict(torch.load('ans_roberta_large_nlock6_bs32/checkpoint-260000'))
	# print('load pretrained model sucessfully ....... ')
	model.to(args.device)
	print('model ' , model)
	logger.info("Training/evaluation parameters %s", args)

	if args.do_train:
		features = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
		global_step, tr_loss = train(args, features, model, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

	#tam thoi comment phan nay 
	#Save# the trained model and the tokenizer
	if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		logger.info("Saving model checkpoint to %s", args.output_dir)

		model.to(args.device)

		# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
		results = {}
		if args.do_eval and args.local_rank in [-1, 0]:

			model.load_state_dict(torch.load('model_draft'))
			model.to(args.device)
			results = evaluate(args, model, tokenizer, prefix=global_step)

		logger.info("Result: {}".format(results))

		return results

	if args.do_eval:
		model.to(args.device)

		# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
		results = {}
		if args.do_eval:

			model.to(args.device)
			results = evaluate(args, model, tokenizer, prefix='not_global_step')

		logger.info("Result: {}".format(results))

		return results

if __name__ == "__main__":
	main()