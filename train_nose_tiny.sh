CUDA_LAUNCH_BLOCKING=1
rm -rf /home/ubuntu/data/processed_video/results_nose_tiny
mkdir /home/ubuntu/data/processed_video/results_nose_tiny
python main_embed.py --root_path /home/ubuntu/data/processed_video \
        --video_path tiny_data_nose_keyp \
        --annotation_path tiny_data_nose_keyp/mh_tiny_binary_fixed.json \
        --result_path results_nose_tiny \
        --dataset mh \
        --n_classes 2 \
        --model embednet \
        --batch_size 6 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 300 \
        --learning_rate 1e-5 \
        --lr_scheduler multistep \
        --sample_duration 352 \
	--weight_decay 1e-4 \
	#--fl_gamma 3 \
        #--momentum 0.5 \