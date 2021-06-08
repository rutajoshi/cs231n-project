CUDA_LAUNCH_BLOCKING=1
rm -rf /home/ubuntu/data/processed_video/results_keypoints
mkdir /home/ubuntu/data/processed_video/results_keypoints
python main_embed.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data_embed \
        --annotation_path binary_data_embed/mh_binary_fixed.json \
        --result_path results_keypoints \
        --dataset mh \
        --n_classes 2 \
        --model embednet \
        --batch_size 5 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 300 \
	--learning_rate 1e-3 \
        --lr_scheduler multistep \
        --sample_duration 352 \
        --weight_decay 1e-4 \
        --train_t_crop center
        #--momentum 0.5 \
	#--fl_gamma 1
