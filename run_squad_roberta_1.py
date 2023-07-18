
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
