#python get_lastlayer_video.py --root_path ~/data/processed_video/ \
#	--video_path phq9_binary_keypoints_3d \
#	--annotation_path phq9_binary_keypoints_3d/mh_01.json \
#	--result_path results_phq9_binary_1fps3d \
#	--dataset mh \
#	--n_classes 2 \
#	--resume_path results_phq9_binary_1fps3d/save_90.pth \
#	--model embednet \
#	--batch_size 16 \
#	--sample_duration 352 \
#	--n_threads 4 \
#	--checkpoint 5 \
#	--inference \
#	--inference_subset all \
#	--no_train \
#	--no_val 

#python get_lastlayer_video.py --root_path ~/data/processed_video/ \
#	--video_path phq9_multiclass_keypoints_3d \
#	--annotation_path phq9_multiclass_keypoints_3d/mh_01.json \
#	--result_path results_phq9_multiclass_1fps3d \
#	--dataset mh \
#	--n_classes 4 \
#	--resume_path results_phq9_multiclass_1fps3d/save_150.pth \
#	--model embednet \
#	--batch_size 16 \
#	--sample_duration 352 \
#	--n_threads 4 \
#	--checkpoint 5 \
#	--inference \
#	--inference_subset all \
#	--no_train \
#	--no_val 

#python get_lastlayer_video.py --root_path ~/data/processed_video/ \
#	--video_path gad7_binary_keypoints_3d \
#	--annotation_path gad7_binary_keypoints_3d/mh_01.json \
#	--result_path results_gad7_binary_1fps3d \
#	--dataset mh \
#	--n_classes 2 \
#	--resume_path results_gad7_binary_1fps3d/save_90.pth \
#	--model embednet \
#	--batch_size 16 \
#	--sample_duration 352 \
#	--n_threads 4 \
#	--checkpoint 5 \
#	--inference \
#	--inference_subset all \
#	--no_train \
#	--no_val 

python get_lastlayer_video.py --root_path ~/data/processed_video/ \
	--video_path gad7_multiclass_keypoints_3d \
	--annotation_path gad7_multiclass_keypoints_3d/mh_01.json \
	--result_path results_gad7_multiclass_1fps3d \
	--dataset mh \
	--n_classes 4 \
	--resume_path results_gad7_multiclass_1fps3d/save_150.pth \
	--model embednet \
	--batch_size 16 \
	--sample_duration 352 \
	--n_threads 4 \
	--checkpoint 5 \
	--inference \
	--inference_subset all \
	--no_train \
	--no_val 

