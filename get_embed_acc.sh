
#python -m util_scripts.eval_accuracy ~/data/processed_video/gad7_multiclass_jpgs/mh_01.json ~/data/processed_video/results_gad7_mul_cenw/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/gad7_multiclass_jpgs/mh_01.json ~/data/processed_video/results_gad7_mul_cenw/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/gad7_binary_jpgs/mh_01.json ~/data/processed_video/results_gad7_bin_cenw/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/gad7_binary_jpgs/mh_01.json ~/data/processed_video/results_gad7_bin_cenw/val.json --subset val -k 1 --ignore

python -m util_scripts.eval_accuracy ~/data/processed_video/question_black/mh_01.json ~/data/processed_video/results_keypoints_mul_cenw/train.json --subset train -k 1 --ignore
python -m util_scripts.eval_accuracy ~/data/processed_video/question_black/mh_01.json ~/data/processed_video/results_keypoints_mul_cenw/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data_embed/mh_01.json ~/data/processed_video/results_keypoints_bin_cenw/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data_embed/mh_01.json ~/data/processed_video/results_keypoints_bin_cenw/val.json --subset val -k 1 --ignore


#python -m util_scripts.eval_accuracy ~/data/processed_video/question_black/mh_01.json ~/data/processed_video/results_keypoints_multiclass/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/question_black/mh_01.json ~/data/processed_video/results_keypoints_multiclass/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data_embed/mh_01.json ~/data/processed_video/results_keypoints_dr06_smallerhidden/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data_embed/mh_01.json ~/data/processed_video/results_keypoints_dr06_smallerhidden/val.json --subset val -k 1 --ignore

