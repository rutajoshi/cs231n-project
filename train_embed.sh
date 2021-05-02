CUDA_LAUNCH_BLOCKING=1
rm -rf /home/ubuntu/data/processed_video/results_embednet
mkdir /home/ubuntu/data/processed_video/results_embednet
python main_embed.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data_embed \
        --annotation_path binary_data_embed/mh_binary_fixed.json \
        --result_path results_embednet \
        --dataset mh \
        --n_classes 2 \
        --model embednet \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 150 \
        --learning_rate 0.0005 \
        --momentum 0.5 \
        --lr_scheduler multistep \
        --sample_duration 224 \
	--fl_gamma 5 \
	--no_cuda
