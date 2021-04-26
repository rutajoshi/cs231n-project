python main.py --root_path /home/ubuntu/data/processed_video \
	--video_path binary_data \
	--annotation_path binary_data/mh_02.json \
	--result_path results_binary_bn_fl0 \
	--dataset mh \
	--n_classes 2 \
	--resume_path results_binary_bn_fl0/save_150.pth \
	--model crnn \
	--model_depth 101 \
	--batch_size 10 \
	--n_threads 4 \
	--checkpoint 5 \
	--inference \
	--inference_subset train \
	--output_topk 1 \
	--inference_batch_size 1 \
	--no_train \
	--no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl0 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl0/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset val \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl05 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl05/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset train \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl05 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl05/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset val \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl10 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl10/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset train \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl10 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl10/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset val \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl20 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl20/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset train \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl20 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl20/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset val \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl50 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl50/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset train \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val

python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl50 \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_binary_bn_fl50/save_150.pth \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset val \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val
