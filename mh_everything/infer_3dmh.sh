#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path phq9_multiclass_keypoints_3d \
#        --annotation_path phq9_multiclass_keypoints_3d/mh_01.json \
#        --result_path results_phq9_multiclass_1fps3d \
#        --dataset mh \
#        --n_classes 4 \
#        --resume_path results_phq9_multiclass_1fps3d/save_90.pth \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --sample_duration 352 \
#        --inference \
#        --inference_subset train \
#        --no_train \
#        --no_val \
#        --mhq_data mh
#
#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path phq9_multiclass_keypoints_3d \
#        --annotation_path phq9_multiclass_keypoints_3d/mh_01.json \
#        --result_path results_phq9_multiclass_1fps3d \
#        --dataset mh \
#        --n_classes 4 \
#        --resume_path results_phq9_multiclass_1fps3d/save_90.pth \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --sample_duration 352 \
#        --inference \
#        --inference_subset val \
#        --no_train \
#        --no_val \
#        --mhq_data mh

python main_embed.py --root_path /home/ubuntu/data/processed_video \
        --video_path phq9_binary_keypoints_3d \
        --annotation_path phq9_binary_keypoints_3d/mh_01.json \
        --result_path results_phq9_binary_1fps3d \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_phq9_binary_1fps3d/save_60.pth \
        --model embednet \
        --batch_size 16 \
        --n_threads 4 \
        --checkpoint 5 \
        --sample_duration 352 \
        --inference \
        --inference_subset train \
        --no_train \
        --no_val \
        --mhq_data mh

python main_embed.py --root_path /home/ubuntu/data/processed_video \
        --video_path phq9_binary_keypoints_3d \
        --annotation_path phq9_binary_keypoints_3d/mh_01.json \
        --result_path results_phq9_binary_1fps3d \
        --dataset mh \
        --n_classes 2 \
        --resume_path results_phq9_binary_1fps3d/save_60.pth \
        --model embednet \
        --batch_size 16 \
        --n_threads 4 \
        --checkpoint 5 \
        --sample_duration 352 \
        --inference \
        --inference_subset val \
        --no_train \
        --no_val \
        --mhq_data mh


#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path tmp_multiclass_3d \
#        --annotation_path tmp_multiclass_3d/mh_01.json \
#        --result_path results_phq9_multiclass_1fps3d_tmp \
#        --dataset mh \
#        --n_classes 4 \
#        --resume_path results_phq9_multiclass_1fps3d_tmp/save_90.pth \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --sample_duration 352 \
#        --inference \
#        --inference_subset train \
#        --no_train \
#        --no_val \
#        --mhq_data mh
#
#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path tmp_multiclass_3d \
#        --annotation_path tmp_multiclass_3d/mh_01.json \
#        --result_path results_phq9_multiclass_1fps3d_tmp \
#        --dataset mh \
#        --n_classes 4 \
#        --resume_path results_phq9_multiclass_1fps3d_tmp/save_90.pth \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --sample_duration 352 \
#        --inference \
#        --inference_subset val \
#        --no_train \
#        --no_val \
#        --mhq_data mh


#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path tmp_binary_3d \
#        --annotation_path tmp_binary_3d/mh_01.json \
#        --result_path results_phq9_binary_1fps3d_tmp \
#        --dataset mh \
#        --n_classes 2 \
#        --resume_path results_phq9_binary_1fps3d_tmp/save_60.pth \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --sample_duration 352 \
#        --inference \
#        --inference_subset train \
#        --no_train \
#        --no_val \
#        --mhq_data mh
#
#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path tmp_binary_3d \
#        --annotation_path tmp_binary_3d/mh_01.json \
#        --result_path results_phq9_binary_1fps3d_tmp \
#        --dataset mh \
#        --n_classes 2 \
#        --resume_path results_phq9_binary_1fps3d_tmp/save_60.pth \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --sample_duration 352 \
#        --inference \
#        --inference_subset val \
#        --no_train \
#        --no_val \
#        --mhq_data mh

#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path fa_binary_3d \
#        --annotation_path fa_binary_3d/mh_01.json \
#        --result_path results_phq9_binary_1fps3d_fa \
#        --dataset mh \
#        --n_classes 2 \
#        --resume_path results_phq9_binary_1fps3d_fa/save_60.pth \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --sample_duration 352 \
#        --inference \
#        --inference_subset train \
#        --no_train \
#        --no_val \
#        --mhq_data mh
#
#python main_embed.py --root_path /home/ubuntu/data/processed_video \
#        --video_path fa_binary_3d \
#        --annotation_path fa_binary_3d/mh_01.json \
#        --result_path results_phq9_binary_1fps3d_fa \
#        --dataset mh \
#        --n_classes 2 \
#        --resume_path results_phq9_binary_1fps3d_fa/save_60.pth \
#        --model embednet \
#        --batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --sample_duration 352 \
#        --inference \
#        --inference_subset val \
#        --no_train \
#        --no_val \
#        --mhq_data mh
