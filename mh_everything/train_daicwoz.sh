
#CUDA_LAUNCH_BLOCKING=1
#rm -rf /home/ubuntu/data/daicwoz_1fps/results_binary
#mkdir /home/ubuntu/data/daicwoz_1fps/results_binary
#python main_embed.py --root_path /home/ubuntu/data/daicwoz_1fps \
#        --video_path daicwoz_binary_keypts_3d \
#        --annotation_path daicwoz_binary_keypts_3d/mh_01.json \
#        --result_path results_binary \
#        --dataset daicwoz \
#        --n_classes 2 \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 30 \
#        --n_epochs 60 \
#        --learning_rate 1e-3 \
#        --lr_scheduler multistep \
#        --sample_duration 400 \
#        --weight_decay 1e-4 \
#        --train_t_crop center \
#        --mhq_data daicwoz \
#        --weighted_sampling_no_norm \
#        #--momentum 0.5 \
#        #--fl_gamma 1

CUDA_LAUNCH_BLOCKING=1
rm -rf /home/ubuntu/data/daicwoz/results_multiclass
mkdir /home/ubuntu/data/daicwoz/results_multiclass
python main_embed.py --root_path /home/ubuntu/data/daicwoz_1fps \
        --video_path daicwoz_multiclass_keypts_3d \
        --annotation_path daicwoz_multiclass_keypts_3d/mh_01.json \
        --result_path results_multiclass \
        --dataset daicwoz \
        --n_classes 4 \
        --model embednet \
        --batch_size 5 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 300 \
        --learning_rate 1e-4 \
        --lr_scheduler multistep \
        --sample_duration 352 \
        --weight_decay 1e-4 \
        --train_t_crop center \
        --mhq_data daicwoz \
        --weighted_sampling_no_norm \
        #--momentum 0.5 \
        #--fl_gamma 1
