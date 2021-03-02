VIDEO_DIR_PATH=/Users/ruta/stanford/pac/mentalhealth/question_clipped/
JPG_DIR_PATH=/Users/ruta/stanford/pac/mentalhealth/question_jpgs/
CSV_DIR_PATH=/Users/ruta/stanford/pac/mentalhealth/CLASSINDPATH
DEST_JSON_PATH=/Users/ruta/stanford/pac/mentalhealth/

VIDEO_DIR_PATH=/share/pi/schul/schul-behavioral/data/3dr_datasets/mh_targets/
JPG_DIR_PATH=/share/pi/schul/schul-behavioral/data/3dr_datasets/mh_video_jpgs/
CSV_DIR_PATH=/share/pi/schul/schul-behavioral/data/3dr_datasets/mh_labels/
DEST_JSON_PATH=/share/pi/schul/schul-behavioral/data/3dr_datasets/

python -m util_scripts.mh_generate_video_jpgs $VIDEO_DIR_PATH $JPG_DIR_PATH mh
python -m util_scripts.mh_json $CSV_DIR_PATH $JPG_DIR_PATH $DEST_JSON_PATH
