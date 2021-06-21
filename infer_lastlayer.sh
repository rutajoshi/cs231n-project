#python get_lastlayer_video.py --root_path ~/data/processed_video/ \
#	--video_path binary_data_embed \
#	--annotation_path binary_data_embed/mh_01.json \
#	--result_path results_keypoints_bin_cenw \
#	--dataset mh \
#	--n_classes 2 \
#	--resume_path results_keypoints_bin_cenw/save_300.pth \
#	--model embednet \
#	--batch_size 5 \
#	--n_threads 4 \
#	--checkpoint 5 \
#	--inference \
#	--inference_subset all \
#	--no_train \
#	--no_val 

#python get_lastlayer_video.py --root_path ~/data/processed_video/ \
#        --video_path question_black \
#        --annotation_path question_black/mh_01.json \
#        --result_path results_keypoints_mul_cenw \
#        --dataset mh \
#        --n_classes 4 \
#        --resume_path results_keypoints_mul_cenw/save_300.pth \
#        --model embednet \
#        --batch_size 5 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --inference \
#        --inference_subset all \
#        --no_train \
#        --no_val

#python get_lastlayer_video.py --root_path ~/data/processed_video/ \
#       --video_path gad7_binary_jpgs \
#       --annotation_path gad7_binary_jpgs/mh_01.json \
#       --result_path results_gad7_bin_cenw \
#       --dataset mh \
#       --n_classes 2 \
#       --resume_path results_gad7_bin_cenw/save_300.pth \
#       --model embednet \
#       --batch_size 5 \
#       --n_threads 4 \
#       --checkpoint 5 \
#       --inference \
#       --inference_subset all \
#       --no_train \
#       --no_val

python get_lastlayer_video.py --root_path ~/data/processed_video/ \
        --video_path gad7_multiclass_jpgs \
        --annotation_path gad7_multiclass_jpgs/mh_01.json \
        --result_path results_gad7_mul_cenw \
        --dataset mh \
        --n_classes 4 \
        --resume_path results_gad7_mul_cenw/save_300.pth \
        --model embednet \
        --batch_size 5 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset all \
        --no_train \
        --no_val
