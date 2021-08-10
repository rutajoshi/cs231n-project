#python saliency2.py --root_path /home/ubuntu/data/processed_video \
#	--video_path phq9_binary_keypoints_3d \
#	--annotation_path phq9_binary_keypoints_3d/mh_01.json \
#	--result_path results_phq9_binary_1fps3d \
#	--dataset mh \
#	--n_classes 2 \
#	--resume_path results_phq9_binary_1fps3d/save_90.pth \
#	--model embednet \
#	--batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 30 \
#        --inference \
#        --inference_subset val \
#        --no_train \
#        --no_val \
# 	--no_cuda

#python saliency2.py --root_path /home/ubuntu/data/processed_video \
#	--video_path phq9_multiclass_keypoints_3d \
#	--annotation_path phq9_multiclass_keypoints_3d/mh_01.json \
#	--result_path results_phq9_multiclass_1fps3d \
#	--dataset mh \
#	--n_classes 4 \
#	--resume_path results_phq9_multiclass_1fps3d/save_150.pth \
#	--model embednet \
#	--batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 30 \
#        --inference \
#        --inference_subset val \
#        --no_train \
#        --no_val \
# 	--no_cuda

python saliency2.py --root_path /home/ubuntu/data/processed_video \
	--video_path gad7_binary_keypoints_3d \
	--annotation_path gad7_binary_keypoints_3d/mh_01.json \
	--result_path results_gad7_binary_1fps3d \
	--dataset mh \
	--n_classes 2 \
	--resume_path results_gad7_binary_1fps3d/save_90.pth \
	--model embednet \
	--batch_size 16 \
        --n_threads 4 \
        --checkpoint 30 \
        --inference \
        --inference_subset val \
        --no_train \
        --no_val \
 	--no_cuda

#python saliency2.py --root_path /home/ubuntu/data/processed_video \
#	--video_path gad7_multiclass_keypoints_3d \
#	--annotation_path gad7_multiclass_keypoints_3d/mh_01.json \
#	--result_path results_gad7_multiclass_1fps3d \
#	--dataset mh \
#	--n_classes 4 \
#	--resume_path results_gad7_multiclass_1fps3d/save_60.pth \
#	--model embednet \
#	--batch_size 16 \
#        --n_threads 4 \
#        --checkpoint 30 \
#        --inference \
#        --inference_subset val \
#        --no_train \
#        --no_val \
# 	--no_cuda
#