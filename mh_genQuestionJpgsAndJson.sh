RAW_VIDEO_DIR_PATH=/home/ubuntu/data/video/
VIDEO_DIR_PATH=/home/ubuntu/data/processed_video/question_cut_videos/
JPG_DIR_PATH=/home/ubuntu/data/processed_video/question_jpgs/
CSV_DIR_PATH=/home/ubuntu/data/processed_video/mhq_local_labels/
DEST_JSON_PATH=/home/ubuntu/data/processed_video/question_cut_videos/
CROPS_DIR_PATH=/home/ubuntu/data/processed_video/question_cropsampled/
ANN_PATH=/home/ubuntu/data/annotation_questions_labeled/
CUT_TARGET_DIR=/home/ubuntu/data/processed_video/mhq_local_targets/

python -m util_scripts.mh_cut_videos $RAW_VIDEO_DIR_PATH $CUT_TARGET_DIR $ANN_PATH mh
#python -m util_scripts.mh_generate_question_video_jpgs $VIDEO_DIR_PATH $JPG_DIR_PATH mh
#python -m util_scripts.mh_crop_and_sample $JPG_DIR_PATH $CROPS_DIR_PATH mh
#python -m util_scripts.mh_json $CSV_DIR_PATH $JPG_DIR_PATH $DEST_JSON_PATH
