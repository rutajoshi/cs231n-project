RAW_VIDEO_DIR_PATH=/home/ubuntu/data/processed_video/vidsbyclass/
CUT_TARGET_DIR=/home/ubuntu/data/processed_video/question_cut_videos/
ANN_PATH=/home/ubuntu/data/annotation_questions_labeled/

JPG_DIR_PATH=/home/ubuntu/data/processed_video/question_jpgs/

CROPS_DIR_PATH=/home/ubuntu/data/processed_video/question_cropsampled/

SAMPLE_DIR_PATH=/home/ubuntu/data/processed_video/question_sampled/
BLACK_DIR_PATH=/home/ubuntu/data/processed_video/question_black/

CSV_DIR_PATH=/home/ubuntu/data/processed_video/mhq_local_labels/
DEST_JSON_PATH=/home/ubuntu/data/processed_video/question_cropsampled/

#python -m util_scripts.mh_cut_videos $RAW_VIDEO_DIR_PATH $CUT_TARGET_DIR $ANN_PATH mh
#python -m util_scripts.mh_generate_question_video_jpgs $CUT_TARGET_DIR $JPG_DIR_PATH mh
#python -m util_scripts.mh_crop_and_sample $JPG_DIR_PATH $CROPS_DIR_PATH mh
python -m util_scripts.mh_json $CSV_DIR_PATH $CROPS_DIR_PATH $DEST_JSON_PATH

#python -m util_scripts.mh_justsample $JPG_DIR_PATH $SAMPLE_DIR_PATH mh
#python -m util_scripts.mh_sampleblack $JPG_DIR_PATH $BLACK_DIR_PATH mh
