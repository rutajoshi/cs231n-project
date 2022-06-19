CUDA_LAUNCH_BLOCKING=1
rm -rf /home/ubuntu/data/processed_video/results_embednet_tiny
mkdir /home/ubuntu/data/processed_video/results_embednet_tiny
python main_embed.py --root_path /home/ubuntu/data/processed_video \
        --video_path tiny_data_embed \
        --annotation_path tiny_data_embed/mh_tiny_binary_fixed.json \
        --result_path results_embednet_tiny \
        --dataset mh \
        --n_classes 2 \
        --model embednet \
        --batch_size 4 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 150 \
        --learning_rate 0.0005 \
        --momentum 0.6 \
        --lr_scheduler plateau \
        --sample_duration 352 \
	--fl_gamma 2
