RAW_VIDEO_DIR_PATH=/Users/ruta/stanford/pac/mentalhealth/local_phi_data/
VIDEO_DIR_PATH=/Users/ruta/stanford/pac/mentalhealth/question_clipped/
JPG_DIR_PATH=/Users/ruta/stanford/pac/mentalhealth/question_jpgs/
CSV_DIR_PATH=/Users/ruta/stanford/pac/mentalhealth/mhq_local_labels/
DEST_JSON_PATH=/Users/ruta/stanford/pac/mentalhealth/question_clipped/
CROPS_DIR_PATH=/Users/ruta/stanford/pac/mentalhealth/question_cropsampled/
ANN_PATH=/Users/ruta/stanford/pac/mentalhealth/local_annotations/
CUT_TARGET_DIR=/Users/ruta/stanford/pac/mentalhealth/mhq_local_targets/

python -m util_scripts.mh_cut_videos $RAW_VIDEO_DIR_PATH $CUT_TARGET_DIR $ANN_PATH mh
#python -m util_scripts.mh_generate_question_video_jpgs $VIDEO_DIR_PATH $JPG_DIR_PATH mh
#python -m util_scripts.mh_crop_and_sample $JPG_DIR_PATH $CROPS_DIR_PATH mh
#python -m util_scripts.mh_json $CSV_DIR_PATH $JPG_DIR_PATH $DEST_JSON_PATH
