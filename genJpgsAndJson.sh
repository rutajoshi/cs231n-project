#python -m util_scripts.generate_video_jpgs ~/data/nturgb/ ~/data/nturgb/jpg/ ntu
#python -m util_scripts.ntu_json ~/ntuTrainTestlist/ ~/data/nturgb/jpg/ ~/data/

#python -m util_scripts.generate_video_jpgs ~/tiny_data/nturgb/ ~/tiny_data/nturgb/jpg/ ntu
#python -m util_scripts.ntu_json ~/tiny_ntuTrainTestlist/ ~/tiny_data/nturgb/jpg/ ~/tiny_data/

python -m util_scripts.generate_video_jpgs ~/all_actions_data/nturgb/ ~/all_actions_data/nturgb/jpg/ ntu
python -m util_scripts.ntu_json ~/all_ntuTrainTestlist/ ~/all_actions_data/nturgb/jpg/ ~/all_actions_data/
