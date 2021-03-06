python saliency.py --root_path ~/teeny_data \
	--video_path nturgb/jpg \
	--annotation_path ntu_01.json \
	--result_path results \
	--dataset ntu \
	--n_classes 9 \
	--resume_path models/big101.pth \
	--model resnet \
	--model_depth 101 \
	--batch_size 10 \
	--inference_batch_size 1 \
	--n_threads 4 \
	--checkpoint 5 \
	--inference_subset val \
	--no_train \
	--no_val 
