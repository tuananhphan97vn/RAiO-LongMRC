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
