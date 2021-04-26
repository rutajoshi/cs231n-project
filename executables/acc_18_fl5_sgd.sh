python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data/mh_02.json ~/data/processed_video/results_bin_18_fl5_sgd/train.json --subset train -k 1 --ignore
python -m util_scripts.eval_accuracy ~/data/processed_video/binary_data/mh_02.json ~/data/processed_video/results_bin_18_fl5_sgd/val.json --subset val -k 1 --ignore

