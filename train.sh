#CUDA_LAUNCH_BLOCKING=1
#rm -rf /home/ubuntu/ruta/results/erus_1
#mkdir /home/ubuntu/ruta/results/erus_1
#python main.py --root_path /home/ubuntu/ruta/data \
#	--video_path jpgs \
#	--annotation_path /home/ubuntu/ruta/labels/split_labels/mh_01.json \
#	--result_path ../results/erus_1 \
#	--dataset erus \
#	--n_classes 4 \
#	--model resnet \
#	--model_depth 101 \
#	--batch_size 50 \
#	--n_threads 4 \
#	--checkpoint 30 \
#	--n_epochs 200 \
#	--learning_rate 1e-2 \
#	--weight_decay 1e-3 \
#	--momentum 0.95 \
#	--lr_scheduler multistep

CUDA_LAUNCH_BLOCKING=1
rm -rf /home/ubuntu/ruta/results/erus_binary
mkdir /home/ubuntu/ruta/results/erus_binary
python main.py --root_path /home/ubuntu/ruta/data \
        --video_path binary_jpgs \
        --annotation_path /home/ubuntu/ruta/labels/binary_labels/mh_01.json \
        --result_path ../results/erus_binary \
        --dataset erus \
        --n_classes 2 \
        --model resnet \
        --model_depth 101 \
        --batch_size 50 \
        --n_threads 4 \
        --checkpoint 30 \
        --n_epochs 200 \
        --learning_rate 1e-3 \
        --weight_decay 1e-3 \
        --momentum 0.95 \
        --lr_scheduler multistep
