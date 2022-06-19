VIDEO_DIR_PATH=/home/ubuntu/ruta/data/videos/
JPG_DIR_PATH=/home/ubuntu/ruta/data/jpgs/
CSV_DIR_PATH=/home/ubuntu/ruta/labels/split_labels/
DEST_JSON_PATH=/home/ubuntu/ruta/labels/split_labels/

#python -m util_scripts.erus_generate_video_jpgs $VIDEO_DIR_PATH $JPG_DIR_PATH erus
python -m util_scripts.erus_json $CSV_DIR_PATH $JPG_DIR_PATH $DEST_JSON_PATH
