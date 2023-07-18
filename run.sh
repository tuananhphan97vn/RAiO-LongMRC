python run_squad_roberta.py \
	--model_type roberta \
	--output_dir nblock1_28_2_2023_s1 \
	--evaluate_during_training \
	--do_lower_case \
	--learning_rate 1e-5 \
	--num_train_epochs 5 \
	--max_seq_length 512\
	--max_query_length 30 \
	--max_answer_length 30\
	--max_len_context 478\
	--per_gpu_eval_batch_size 1 \
	--gradient_accumulation_steps 32 \
	--per_gpu_train_batch_size 1 \
	--logging_steps 5000 \
	--save_steps 2000 \
	--seed 1 \
	--threads 16 \
	--train_file  /storage-nlp/public-dataset/newsqa/train.json \
	--model_name_or_path roberta-base \
	--do_train\
	--overwrite_output_dir\
	--cache_dir ./cache \
	--predict_file  /storage-nlp/public-dataset/newsqa/dev.json \
	--n_block 0 \
	--n_block_reasoning 1 --warmup_steps 1000 \
	--word_dim 768 \
	--sent_dim 768 --chunk_length 0 --version_2_with_negative
	#--tokenizer_name ~/work/huggingface/vinai/phobert-base \
	# --config_name ~/work/huggingface/vinai/phobert-base \
	# --eval_all_checkpoint \
	#--version_2_with_negative
	# --null_score_diff_threshold 0.0
	#--predict_file /data/nlp/hoangnv74/question-answering/data/viquad/test_positive.json \
	# --do_eval \

