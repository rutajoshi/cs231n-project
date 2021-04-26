rm -rf /home/ubuntu/data/processed_video/results_nopretrain_34fl5
mkdir /home/ubuntu/data/processed_video/results_nopretrain_34fl5
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path question_cropsampled \
        --annotation_path question_cropsampled/mh_01.json \
        --result_path results_nopretrain_34fl5 \
        --dataset mh \
        --n_classes 4 \
        --model crnn \
        --model_depth 34 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 150 \
        --learning_rate 0.0005 \
        --momentum 0.5 \
        --lr_scheduler multistep \
        --sample_duration 224 \
	--fl_gamma 5
