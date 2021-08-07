
#CUDA_LAUNCH_BLOCKING=1
#rm -rf /home/ubuntu/data/processed_video/results_phq9_binary_1fps3d_fa
#mkdir /home/ubuntu/data/processed_video/results_phq9_binary_1fps3d_fa
#python main_embed.py --root_path /home/ubuntu/data/processed_video/ \
#        --video_path fa_binary_3d \
#        --annotation_path fa_binary_3d/mh_01.json \
#        --result_path results_phq9_binary_1fps3d_fa \
#        --dataset mh \
#        --n_classes 2 \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 20 \
#        --n_epochs 100 \
#        --learning_rate 5e-4 \
#        --lr_scheduler multistep \
#        --sample_duration 352 \
#        --weight_decay 1e-4 \
#        --train_t_crop center \
#        --mhq_data phq9 \
#        --weighted_sampling_norm \
#        #--momentum 0.25 \
#        #--fl_gamma 1

#CUDA_LAUNCH_BLOCKING=1
#rm -rf /home/ubuntu/data/processed_video/results_phq9_multiclass_1fps3d_fa
#mkdir /home/ubuntu/data/processed_video/results_phq9_multiclass_1fps3d_fa
#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path fa_multiclass_3d \
#        --annotation_path fa_multiclass_3d/mh_01.json \
#        --result_path results_phq9_multiclass_1fps3d_fa \
#        --dataset mh \
#        --n_classes 4 \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 30 \
#        --n_epochs 100 \
#        --learning_rate 1e-4 \
#        --lr_scheduler multistep \
#        --sample_duration 352 \
#        --weight_decay 1e-4 \
#        --train_t_crop center \
#        --mhq_data phq9 \
#        #--weighted_sampling_no_norm \
#        #--momentum 0.5 \
#        #--fl_gamma 1

#CUDA_LAUNCH_BLOCKING=1
#rm -rf /home/ubuntu/data/processed_video/results_phq9_binary_1fps3d_tmp
#mkdir /home/ubuntu/data/processed_video/results_phq9_binary_1fps3d_tmp
#python main_embed.py --root_path /home/ubuntu/data/processed_video/ \
#        --video_path tmp_binary_3d \
#        --annotation_path tmp_binary_3d/mh_01.json \
#        --result_path results_phq9_binary_1fps3d_tmp \
#        --dataset mh \
#        --n_classes 2 \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 20 \
#        --n_epochs 100 \
#        --learning_rate 5e-4 \
#        --lr_scheduler multistep \
#        --sample_duration 352 \
#        --weight_decay 1e-4 \
#        --train_t_crop center \
#        --mhq_data phq9 \
#        --weighted_sampling_norm \
#        --momentum 0.25 \
#        #--fl_gamma 1

#CUDA_LAUNCH_BLOCKING=1
#rm -rf /home/ubuntu/data/processed_video/results_phq9_multiclass_1fps3d_tmp
#mkdir /home/ubuntu/data/processed_video/results_phq9_multiclass_1fps3d_tmp
#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path tmp_multiclass_3d \
#        --annotation_path tmp_multiclass_3d/mh_01.json \
#        --result_path results_phq9_multiclass_1fps3d_tmp \
#        --dataset mh \
#        --n_classes 4 \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 30 \
#        --n_epochs 100 \
#        --learning_rate 1e-4 \
#        --lr_scheduler multistep \
#        --sample_duration 352 \
#        --weight_decay 1e-4 \
#        --train_t_crop center \
#        --mhq_data phq9 \
#        #--weighted_sampling_no_norm \
#        #--momentum 0.5 \
#        #--fl_gamma 1

#CUDA_LAUNCH_BLOCKING=1
#rm -rf /home/ubuntu/data/processed_video/results_phq9_binary_1fps3d
#mkdir /home/ubuntu/data/processed_video/results_phq9_binary_1fps3d
#python main_embed.py --root_path /home/ubuntu/data/processed_video/ \
#        --video_path phq9_binary_keypoints_3d \
#        --annotation_path phq9_binary_keypoints_3d/mh_01.json \
#        --result_path results_phq9_binary_1fps3d \
#        --dataset mh \
#        --n_classes 2 \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 30 \
#        --n_epochs 100 \
#        --learning_rate 1e-4 \
#        --lr_scheduler multistep \
#        --sample_duration 352 \
#        --weight_decay 1e-4 \
#        --train_t_crop center \
#        --mhq_data phq9 \
#        --weighted_sampling_norm \
#        #--momentum 0.5 \
#        #--fl_gamma 1

CUDA_LAUNCH_BLOCKING=1
rm -rf /home/ubuntu/data/processed_video/results_phq9_multiclass_1fps3d
mkdir /home/ubuntu/data/processed_video/results_phq9_multiclass_1fps3d
python main_embed.py --root_path /home/ubuntu/data/processed_video \
        --video_path phq9_multiclass_keypoints_3d \
        --annotation_path phq9_multiclass_keypoints_3d/mh_01.json \
        --result_path results_phq9_multiclass_1fps3d \
        --dataset mh \
        --n_classes 4 \
        --model embednet \
        --batch_size 16 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 100 \
        --learning_rate 1e-4 \
        --lr_scheduler multistep \
        --sample_duration 352 \
        --weight_decay 1e-4 \
        --train_t_crop center \
        --mhq_data phq9 \
        #--weighted_sampling_no_norm \
        #--momentum 0.5 \
        #--fl_gamma 1
