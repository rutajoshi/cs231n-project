
python -m util_scripts.eval_accuracy ~/data/processed_video/phq9_binary_keypoints_3d/mh_01.json ~/data/processed_video/results_phq9_binary_1fps3d/train.json --subset train -k 1 --ignore
python -m util_scripts.eval_accuracy ~/data/processed_video/phq9_binary_keypoints_3d/mh_01.json ~/data/processed_video/results_phq9_binary_1fps3d/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/phq9_multiclass_keypoints_3d/mh_01.json ~/data/processed_video/results_phq9_multiclass_1fps3d/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/phq9_multiclass_keypoints_3d/mh_01.json ~/data/processed_video/results_phq9_multiclass_1fps3d/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/daicwoz_1fps/daicwoz_binary_keypts_3d/mh_01.json ~/data/daicwoz_1fps/results_binary/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/daicwoz_1fps/daicwoz_binary_keypts_3d/mh_01.json ~/data/daicwoz_1fps/results_binary/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/daicwoz/daicwoz_binary_keypts/mh_01.json ~/data/daicwoz/results_binary/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/daicwoz/daicwoz_binary_keypts/mh_01.json ~/data/daicwoz/results_binary/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/gad7_multiclass_jpgs/mh_01.json ~/data/processed_video/results_gad7_mul_cenw/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/gad7_multiclass_jpgs/mh_01.json ~/data/processed_video/results_gad7_mul_cenw/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/gad7_binary_jpgs/mh_01.json ~/data/processed_video/results_gad7_bin_cenw/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/gad7_binary_jpgs/mh_01.json ~/data/processed_video/results_gad7_bin_cenw/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/question_black/mh_01.json ~/data/processed_video/results_keypoints_mul_cenw/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/question_black/mh_01.json ~/data/processed_video/results_keypoints_mul_cenw/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data_embed/mh_01.json ~/data/processed_video/results_keypoints_bin_cenw/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data_embed/mh_01.json ~/data/processed_video/results_keypoints_bin_cenw/val.json --subset val -k 1 --ignore




### --- old models that did "well" (multiclass phq9 did around 30%) ---- ###

#python -m util_scripts.eval_accuracy ~/data/processed_video/question_black/mh_01.json ~/data/processed_video/results_keypoints_multiclass/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/question_black/mh_01.json ~/data/processed_video/results_keypoints_multiclass/val.json --subset val -k 1 --ignore

#python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data_embed/mh_01.json ~/data/processed_video/results_keypoints_dr06_smallerhidden/train.json --subset train -k 1 --ignore
#python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data_embed/mh_01.json ~/data/processed_video/results_keypoints_dr06_smallerhidden/val.json --subset val -k 1 --ignore

