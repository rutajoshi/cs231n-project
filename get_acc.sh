python -m util_scripts.eval_accuracy ~/ruta/labels/binary_labels/mh_01.json ~/ruta/results/erus_binary/train.json --subset train -k 1 --ignore
python -m util_scripts.eval_accuracy ~/ruta/labels/binary_labels/mh_01.json ~/ruta/results/erus_binary/val.json --subset val -k 1 --ignore
python -m util_scripts.eval_accuracy ~/ruta/labels/binary_labels/mh_01.json ~/ruta/results/erus_binary/test.json --subset test -k 1 --ignore
