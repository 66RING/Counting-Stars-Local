cs count_star:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python ./test.py --model $(MODELS_DIR)/tinyllama-110M

