mkdir /home/ubuntu/data/processed_video/results_18_m90
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path question_cropsampled \
        --annotation_path question_cropsampled/mh_01.json \
        --result_path results_18_m90 \
        --dataset mh \
        --n_classes 4 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d18_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 18 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 150 \
        --learning_rate 0.0005 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224

mkdir /home/ubuntu/data/processed_video/results_34_m90
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path question_cropsampled \
        --annotation_path question_cropsampled/mh_01.json \
        --result_path results_34_m90 \
        --dataset mh \
        --n_classes 4 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d34_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 34 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 150 \
        --learning_rate 0.0005 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224

mkdir /home/ubuntu/data/processed_video/results_50_m90
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path question_cropsampled \
        --annotation_path question_cropsampled/mh_01.json \
        --result_path results_50_m90 \
        --dataset mh \
        --n_classes 4 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d50_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 50 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 150 \
        --learning_rate 0.0005 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224

mkdir /home/ubuntu/data/processed_video/results_101_m90
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path question_cropsampled \
        --annotation_path question_cropsampled/mh_01.json \
        --result_path results_101_m90 \
        --dataset mh \
        --n_classes 4 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d101_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 101 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 150 \
        --learning_rate 0.0005 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224

mkdir /home/ubuntu/data/processed_video/results_152_m90
python main.py --root_path /home/ubuntu/data/processed_video \
        --video_path question_cropsampled \
        --annotation_path question_cropsampled/mh_01.json \
        --result_path results_152_m90 \
        --dataset mh \
        --n_classes 4 \
        --n_pretrain_classes 1039 \
        --pretrain_path /home/ubuntu/ruta/pretrained/r3d152_KM_200ep.pth \
        --ft_begin_module fc \
        --model crnn \
        --model_depth 152 \
        --batch_size 10 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 150 \
        --learning_rate 0.0005 \
        --weight_decay 1e-3 \
        --momentum 0.9 \
        --lr_scheduler plateau \
        --sample_duration 224
