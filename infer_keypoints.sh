#python main_embed.py --root_path ~/data/processed_video/ \
#	--video_path phq9_binary_jpgs \
#	--annotation_path phq9_binary_jpgs/mh_01.json \
#	--result_path results_keypoints_bin_cenw \
#	--dataset mh \
#	--n_classes 2 \
#	--resume_path results_keypoints_bin_cenw/save_300.pth \
#	--model embednet \
#	--batch_size 5 \
#	--n_threads 4 \
#	--checkpoint 5 \
#	--inference \
#	--inference_subset train \
#	--no_train \
#	--no_val \
#	--mhq_data phq9
#
#python main_embed.py --root_path ~/data/processed_video/ \
#	--video_path phq9_binary_jpgs \
#	--annotation_path phq9_binary_jpgs/mh_01.json \
#	--result_path results_keypoints_bin_cenw \
#	--dataset mh \
#	--n_classes 2 \
#	--resume_path results_keypoints_bin_cenw/save_300.pth \
#	--model embednet \
#	--batch_size 5 \
#	--n_threads 4 \
#	--checkpoint 5 \
#	--inference \
#	--inference_subset val \
#	--no_train \
#	--no_val \
#	--mhq_data phq9

#python main_embed.py --root_path ~/data/processed_video/ \
#        --video_path phq9_multiclass_jpgs \
#        --annotation_path phq9_multiclass_jpgs/mh_01.json \
#        --result_path results_keypoints_mul_cenw \
#        --dataset mh \
#        --n_classes 4 \
#        --resume_path results_keypoints_mul_cenw/save_300.pth \
#        --model embednet \
#        --batch_size 5 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --inference \
#        --inference_subset train \
#        --no_train \
#        --no_val \
#	 --mhq_data phq9
#
#python main_embed.py --root_path ~/data/processed_video/ \
#        --video_path phq9_multiclass_jpgs \
#        --annotation_path phq9_multiclass_jpgs/mh_01.json \
#        --result_path results_keypoints_mul_cenw \
#        --dataset mh \
#        --n_classes 4 \
#        --resume_path results_keypoints_mul_cenw/save_300.pth \
#        --model embednet \
#        --batch_size 5 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --inference \
#        --inference_subset val \
#        --no_train \
#        --no_val \
#	 --mhq_data phq9

#python main_embed.py --root_path ~/data/processed_video/ \
#        --video_path gad7_binary_jpgs \
#        --annotation_path gad7_binary_jpgs/mh_01.json \
#        --result_path results_gad7_bin_cenw \
#        --dataset mh \
#        --n_classes 2 \
#        --resume_path results_gad7_bin_cenw/save_300.pth \
#        --model embednet \
#        --batch_size 5 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --inference \
#        --inference_subset train \
#        --no_train \
#        --no_val \
#	 --mhq_data gad7
#
#python main_embed.py --root_path ~/data/processed_video/ \
#        --video_path gad7_binary_jpgs \
#        --annotation_path gad7_binary_jpgs/mh_01.json \
#        --result_path results_gad7_bin_cenw \
#        --dataset mh \
#        --n_classes 2 \
#        --resume_path results_gad7_bin_cenw/save_300.pth \
#        --model embednet \
#        --batch_size 5 \
#        --n_threads 4 \
#        --checkpoint 5 \
#        --inference \
#	--inference_subset val \
#        --no_train \
#        --no_val \
#	 --mhq_data gad7

python main_embed.py --root_path ~/data/processed_video/ \
        --video_path gad7_multiclass_jpgs \
        --annotation_path gad7_multiclass_jpgs/mh_01.json \
        --result_path results_gad7_mul_cenw \
        --dataset mh \
        --n_classes 4 \
        --resume_path results_gad7_mul_cenw/save_120.pth \
        --model embednet \
        --batch_size 5 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset train \
        --no_train \
        --no_val \
	--mhq_data gad7

python main_embed.py --root_path ~/data/processed_video/ \
        --video_path gad7_multiclass_jpgs \
        --annotation_path gad7_multiclass_jpgs/mh_01.json \
        --result_path results_gad7_mul_cenw \
        --dataset mh \
        --n_classes 4 \
        --resume_path results_gad7_mul_cenw/save_120.pth \
        --model embednet \
        --batch_size 5 \
        --n_threads 4 \
        --checkpoint 5 \
        --inference \
        --inference_subset val \
        --no_train \
        --no_val \
	--mhq_data gad7
