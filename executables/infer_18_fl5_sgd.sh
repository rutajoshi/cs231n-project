python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_bin_18_fl5_adam \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_bin_18_fl5_adam/save_100.pth \
        --model crnn \
        --model_depth 18 \
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
        --result_path results_bin_18_fl5_adam \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_bin_18_fl5_adam/save_100.pth \
        --model crnn \
        --model_depth 18 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset val \
        --output_topk 1 \
        --inference_batch_size 1 \
        --no_train \
        --no_val