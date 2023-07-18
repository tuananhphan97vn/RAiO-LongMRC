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
