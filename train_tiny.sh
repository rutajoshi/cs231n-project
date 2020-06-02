python main.py --root_path ~/tiny_data \
	--video_path nturgb/jpg \
	--annotation_path ntu_01.json \
	--result_path results \
	--dataset ntu \
	--n_classes 9 \
	--n_pretrain_classes 1039 \
	--pretrain_path models/r3d101_KM_200ep.pth \
	--ft_begin_module fc \
	--model resnet \
	--model_depth 101 \
	--batch_size 10 \
	--n_threads 4 \
	--checkpoint 5 \
	--inference \
	--inference_subset val \
	--n_epochs 10 \
	--learning_rate 0.001 \
	--weight_decay 1e-3 \
	--lr_scheduler plateau
