#CUDA_LAUNCH_BLOCKING=1
#rm -rf /home/ubuntu/data/processed_video/results_keypoints_bin_cenw
#mkdir /home/ubuntu/data/processed_video/results_keypoints_bin_cenw
#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path binary_data_embed \
#        --annotation_path binary_data_embed/mh_01.json \
#        --result_path results_keypoints_bin_cenw \
#        --dataset mh \
#        --n_classes 2 \
#        --model embednet \
#        --batch_size 5 \
#        --n_threads 4 \
#        --checkpoint 30 \
#        --n_epochs 150 \
#	--learning_rate 1e-3 \
#        --lr_scheduler multistep \
#        --sample_duration 352 \
#        --weight_decay 1e-4 \
#        --train_t_crop center
#        #--momentum 0.5 \
#	#--fl_gamma 1

CUDA_LAUNCH_BLOCKING=1
rm -rf /home/ubuntu/data/processed_video/results_keypoints_mul_cenw
mkdir /home/ubuntu/data/processed_video/results_keypoints_mul_cenw
python main_embed.py --root_path /home/ubuntu/data/processed_video \
        --video_path question_black \
        --annotation_path question_black/mh_01.json \
        --result_path results_keypoints_mul_cenw \
        --dataset mh \
        --n_classes 4 \
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

