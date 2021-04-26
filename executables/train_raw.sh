python main.py --root_path ~/data \
	--video_path nturgb/jpg \
	--annotation_path ntu_01.json \
	--result_path results \
	--dataset ntu \
	--n_classes 9 \
	--ft_begin_module fc \
	--model resnet \
	--model_depth 101 \
	--batch_size 80 \
	--n_threads 4 \
	--checkpoint 5 \
	--inference \
	--inference_subset val \
	--n_epochs 200 \
	--learning_rate 0.005 \
	#--lr_scheduler plateau
