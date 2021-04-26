mkdir /home/ubuntu/data/processed_video/results_binary_bn_fl05
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl05 \
        --dataset mh \
        --n_classes 2 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d101_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 25 \
        --n_epochs 150 \
        --learning_rate 0.001 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224 \
        --fl_gamma 0.5

mkdir /home/ubuntu/data/processed_video/results_binary_bn_fl10
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl10 \
        --dataset mh \
        --n_classes 2 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d101_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 25 \
        --n_epochs 150 \
        --learning_rate 0.001 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224 \
        --fl_gamma 1.0

mkdir /home/ubuntu/data/processed_video/results_binary_bn_fl20
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl20 \
        --dataset mh \
        --n_classes 2 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d101_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 25 \
        --n_epochs 150 \
        --learning_rate 0.001 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224 \
        --fl_gamma 2.0

mkdir /home/ubuntu/data/processed_video/results_binary_bn_fl50
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path binary_data \
        --annotation_path binary_data/mh_02.json \
        --result_path results_binary_bn_fl50 \
        --dataset mh \
        --n_classes 2 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d101_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 25 \
        --n_epochs 150 \
        --learning_rate 0.001 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224 \
        --fl_gamma 5.0
