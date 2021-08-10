from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler, Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision

from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop, TemporalBeginCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset_embed import get_training_data, get_validation_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
from validation import val_epoch
import inference

# ADDED for 231n
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from focalloss import FocalLoss, compute_class_weight

#ACTION_NAMES = ["minimal", "mildLow", "modMedium", "severeHigh"]
#ACTION_DICT = {"minimal" : 0, "mildLow" : 1, "modMedium" : 2, "severeHigh" : 3}
ACTION_NAMES = ["minimal", "notable"]
ACTION_DICT = {"minimal" : 0, "notable" : 1}

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt


def get_mse_labels(opt):
    label_dict = {}

    if not opt.csv_path:
        return None

    # Read the labels csv into a df so that you can read the files in numerical order
    csv_df = pd.read_csv(opt.csv_path)

    # For each line in the csv, skipping the header, find the corresponding video file
    for index, row in csv_df.iterrows():
        patient_id = int(row['participant_id'])
        score = int(row['PHQ9_score'])
        if opt.mhq_data == 'gad7':
            score = int(row['GAD7_score'])

        video_id = "zoom_" + str(patient_id)
        if patient_id < 34:
            video_id = "inperson_" + str(patient_id)

        label_dict[video_id] = score

    print("Made label dict: " + str(len(label_dict)) + " labels")

    return label_dict

def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    print("Arch = " + str(arch))
    print("Checkpoint arch = " + str(checkpoint['arch']))
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, model_parameters):
    assert opt.train_crop in ['random', 'corner', 'center']
    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    #if opt.sample_t_stride > 1:
    #    temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    #if opt.train_t_crop == 'random':
    #    temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    #elif opt.train_t_crop == 'center':
    #    print("CROPPING WITH CENTER")
    #    temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform.append(TemporalBeginCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    mse_labels = get_mse_labels(opt)

    train_data = get_training_data(opt.video_path, opt.annotation_path,
                                   opt.dataset, opt.input_type, opt.file_type,
                                   None, temporal_transform, None, mse_labels)

    print("Size of train data = " + str(len(train_data)))
    print("opt.batch_size = " + str(opt.batch_size))

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
    else:
        train_sampler = None

    if opt.weighted_sampling_no_norm:
        # Weighted sampler without normalizing
        all_labels = [x[1] for x in train_data]
        class_counts = [0 for i in range(opt.n_classes)]
        for label in all_labels:
            class_counts[label] += 1
        N = sum(class_counts)
        weight_per_class = [0.] * opt.n_classes
        for i in range(opt.n_classes):
            weight_per_class[i] = N / float(class_counts[i])
        #manual = [3.0, 1.0, 0.8, 0.9]
        weights = [0] * len(train_data)
        for idx, val in enumerate(train_data):
            weights[idx] = weight_per_class[val[1]] #* manual[val[1]]
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    if opt.weighted_sampling_norm:
        # Weighted sampler - same as CE loss weights (with normalizing)
        label_path = "/home/ubuntu/data/processed_video/phq9_binary_labels/trainlist01.txt"
        if opt.mhq_data == "gad7":
            label_path = "/home/ubuntu/data/processed_video/gad7_binary_labels/trainlist01.txt"
        elif opt.mhq_data == "daicwoz":
            label_path = "/home/ubuntu/data/daicwoz/daicwoz_binary_labels/trainlist01.txt"

        if opt.n_classes == 4:
            label_path = "/home/ubuntu/data/processed_video/phq9_multiclass_labels/trainlist01.txt"
            if opt.mhq_data == "gad7":
                label_path = "/home/ubuntu/data/processed_video/gad7_multiclass_labels/trainlist01.txt"
            elif opt.mhq_data == "daicwoz":
                label_path = "/home/ubuntu/data/daicwoz/daicwoz_multiclass_labels/trainlist01.txt"

        if opt.label_path is not None:
            label_path = opt.label_path
        labels = []
        with open(label_path, 'r') as f:
            labels = torch.IntTensor([int(line.split(" ")[1]) for line in f])
        if (len(labels) == 0):
            print("LABELS IS EMPTY")
        weight_per_class = compute_class_weight(labels, opt.n_classes)
        weights = [0] * len(train_data)
        for idx, val in enumerate(train_data):
            weights[idx] = weight_per_class[val[1]]
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    #optimizer = SGD(model_parameters,
    #                lr=opt.learning_rate,
    #                momentum=opt.momentum,
    #                dampening=dampening,
    #                weight_decay=opt.weight_decay,
    #                nesterov=opt.nesterov)

    # ADDED for CS231n
    optimizer = Adam(model_parameters,
                     lr=opt.learning_rate,
                     betas=(0.9, 0.999),
                     eps=1e-08,
                     weight_decay=opt.weight_decay)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        print("Multistep milestones = " + str(opt.multistep_milestones))
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones, verbose=True)

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)

    temporal_transform = []
    temporal_transform.append(TemporalBeginCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    #val_data, collate_fn = get_validation_data(opt.video_path,
    #                                           opt.annotation_path, opt.dataset,
    #                                           opt.input_type, opt.file_type,
    #                                           spatial_transform,
    #                                           temporal_transform)

    mse_labels = get_mse_labels(opt)

    val_data, collate_fn = get_validation_data(opt.video_path,
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,
                                               None,
                                               temporal_transform, None, mse_labels)

    print("Size of val data = " + str(len(val_data)))
    print("opt.batch_size = " + str(opt.batch_size))
    print("opt.n_val_samples = " + str(opt.n_val_samples))

    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size //
                                                         opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn)#,
                                             #collate_fn=collate_fn)

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None

    return val_loader, val_logger


def get_inference_utils(opt):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    temporal_transform = []
    temporal_transform.append(TemporalBeginCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    #inference_data, collate_fn = get_inference_data(
    #    opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
    #    opt.file_type, opt.inference_subset, spatial_transform,
    #    temporal_transform)

    mse_labels = get_mse_labels(opt)

    inference_data, collate_fn = get_inference_data(
        opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
        opt.file_type, opt.inference_subset, None,
        temporal_transform, None, mse_labels)

    print("Got inference data")

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn) #,
        #collate_fn=collate_fn)

    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)

def get_saliency_map(X, y, model, opt):
    """
    This is a function added for 231n.
    Compute a class saliency map using the model for single video X and label y.

    Input:
    - X: Input video; Tensor of shape (1, 3, T, H, W) -- 1 video, 3 channels, T frames, HxW images
    - y: Labels for X; LongTensor of shape (1,) -- 1 label
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (1, T, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensors require gradient
    print("X shape = " + str(X.shape)) #204 for 3d, 136 for 2d
    X.requires_grad_()
    saliency = None

    # Get the labels as a json dict
    with opt.annotation_path.open('r') as f:
        data = json.load(f)

    # Convert y (targets) into labels
    labels = []
    for elem in y:
        #label = int(elem[0].split('_')[0][-2:]) - 41
        #print(elem) # ['inperson_30', [1, 353]] --> use the number to get the label
        label = data['database'][elem[0]]["annotations"]["label"]
        label = ACTION_DICT[label]
        labels.append(label)

    y = torch.LongTensor(labels).to(opt.device)
    scores = model(X).gather(1, y.view(-1, 1)).squeeze().sum()
    scores.backward()
    saliency, temp = X.grad.data.abs().max(dim = 1)
    print("saliency = " + str(saliency.shape))
    return saliency

def plot_saliency_3d(sal_map, i, inputs, targets, opt):
    # Use matplotlib to make one figure showing the average image for each segment
    # for the video and the saliency map for each segment of the video

    # For a video with 5 segments which results in sal_map 5x16x112x112
    # We avg over the 16 saliency maps (one for each image in the segment) to get 5x112x112
    # inputs has shape 5x3x16x112x112 --> this is the segment of input images
    # Avg over 16 images in the segment and take max over 3 channels of each image
    # Plot each of the 5 images with corresponding heatmap of saliency

    # --- MH ---
    # saliency map has shape 5x351x204 --> 5 is batch size, 351 images per video, each video gets a 1x204 keypoint vector
    # y is the labels --> shape 5
    # Cut the input into 5 segments manually --> 5x70x5x204
    # average over 70 images to get 5x5x204
    # plot those 5 images from the original video, with the keypoints overlayed

    # Get the labels as a json dict
    with opt.annotation_path.open('r') as f:
        data = json.load(f)

    with torch.no_grad():
        sal_map = sal_map.numpy()
        print("Original sal map shape = " + str(sal_map.shape))
        # 1. cut the saliency map into 5 segments
        # 1. Average over saliency map dimensions
        sal_map = np.expand_dims(sal_map, axis=1)[:,:,:-1,:] # remove 351st image
        sal_shape = sal_map.shape
        print("Sal shape = " + str(sal_shape)) # 13x1x350x204
        #sal_map = np.reshape(sal_map, (sal_shape[0], sal_shape[2]//5, 5, sal_shape[3])) # should be 13x70x5x204
        beginning = np.reshape(sal_map[:,:,:320, :], (sal_shape[0], 32, 10, sal_shape[3])) # 13x32x10x204
        end = np.reshape(sal_map[:, :, 320:, :], (13, 30, 1, 204)) # 13x30x1x204

        # Average over each segment
        #avg_sal_map = np.mean(sal_map, axis=1) # 13x5x204
        avg_sal_map = np.max(beginning, axis=1) #np.mean(beginning, axis=1) #13x10x204
        end_avg = np.max(end, axis=1) #np.mean(end, axis=1) #13x1x204
        avg_sal_map = np.concatenate((avg_sal_map, end_avg), axis=1) #13x11x204

        # 3. Convert targets into labels
        labels = []
        for elem in targets:
            label = data['database'][elem[0]]["annotations"]["label"]
            label = ACTION_DICT[label]
            labels.append(label)
        y = torch.LongTensor(labels).to(opt.device)
        print("y shape = " + str(y.shape))

        # For each video, find the 11 relevant images (img [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320])
        #relevant_indices = [i * (sal_shape[2]//11) for i in range(11)]
        relevant_indices = [i * 32 for i in range(11)]
        image_name_formatter = lambda x: f'image_{x:05d}.pt' #keypoint path
        image_names = [image_name_formatter(i) for i in relevant_indices]
        print("image names = " + str(image_names))

        #img_root_dir = "/home/ubuntu/data/processed_video/binary_data_embed"
        #kpt_root_dir = "/home/ubuntu/data/processed_video/phq9_binary_keypoints_3d"
        
        kpt_root_dir = "/home/ubuntu/data/processed_video/gad7_binary_keypoints"
        #kpt_root_dir = "/home/ubuntu/data/processed_video/gad7_multiclass_keypoints"
        #kpt_root_dir = "/home/ubuntu/data/processed_video/phq9_binary_keypoints"
        #kpt_root_dir = "/home/ubuntu/data/processed_video/phq9_multiclass_keypoints"
        
        # For each element in targets, get the relevant image paths and keypoint paths
        # For each relevant image, get the saliency map
        # Plot the image in the background, then scatter the keypoints using saliency as heatmap color
        for i in range(len(targets)):
            elem = targets[i]
            videoname = elem[0]
            classname = data['database'][videoname]["annotations"]["label"]
            #img_dirpath = img_root_dir + "/" + classname + "/" + videoname
            kpt_dirpath = kpt_root_dir + "/" + classname + "/" + videoname
            for j in range(len(image_names)):
                img_name = image_names[j]
                kpt_path = kpt_dirpath + "/" + img_name

                # Get keypoints
                keypoints = torch.load(kpt_path).detach()
                keypoints = torch.reshape(keypoints, (68, 2)).type(torch.FloatTensor) 
                keypoints = keypoints.numpy()
                #print("keypoint shape = " + str(keypoints.shape))
                #print("keypoints = " + str(keypoints))

                # Get saliency map
                sals = avg_sal_map[i][j] # numpy array of length 204
                sals = np.reshape(sals, (68, 3))
                sals = np.mean(sals, axis=1) # shape is (68,) # average over RGB channels of the image
                #print("sals shape = " + str(sals.shape))

                # Plot 3D
                #fig = plt.figure()
                #ax = Axes3D(fig)
                #ax.scatter(keypoints[:,0], keypoints[:,1], s=5)

                # Plot 2D
                #plt.subplot(2, len(image_names), j+1)
                #left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                plt.figure()
                plt.scatter(keypoints[:,0], -keypoints[:,1], s=5, c=sals, cmap=plt.cm.hot)
                plt.axis("off")

                figpath = Path('/home/ubuntu/data/processed_video/salmaps_max/bin_gad/map_' + classname + "_" + videoname + "_" + img_name.split(".")[0] + ".jpg")
                #figpath = Path('/home/ubuntu/data/processed_video/salmaps_max/mul_gad/map_' + classname + "_" + videoname + "_" + img_name.split(".")[0] + ".jpg")
                #figpath = Path('/home/ubuntu/data/processed_video/salmaps_max/bin_phq/map_' + classname + "_" + videoname + "_" + img_name.split(".")[0] + ".jpg")
                #figpath = Path('/home/ubuntu/data/processed_video/salmaps_max/mul_phq/map_' + classname + "_" + videoname + "_" + img_name.split(".")[0] + ".jpg")
                plt.savefig(figpath)
    return None


def plot_saliency_2d(sal_map, i, inputs, targets, opt):
    # Use matplotlib to make one figure showing the average image for each segment
    # for the video and the saliency map for each segment of the video

    # For a video with 5 segments which results in sal_map 5x16x112x112
    # We avg over the 16 saliency maps (one for each image in the segment) to get 5x112x112
    # inputs has shape 5x3x16x112x112 --> this is the segment of input images
    # Avg over 16 images in the segment and take max over 3 channels of each image
    # Plot each of the 5 images with corresponding heatmap of saliency

    # --- MH ---
    # saliency map has shape 5x351x136/204 --> 5 is batch size, 351 images per video, each video gets a 1x136/204 keypoint vector
    # y is the labels --> shape 5
    # Cut the input into 5 segments manually --> 5x70x5x136
    # average over 70 images to get 5x5x136
    # plot those 5 images from the original video, with the keypoints overlayed

    # Get the face detector and predictor from dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("util_scripts/shape_predictor_68_face_landmarks.dat")

    # Get the labels as a json dict
    with opt.annotation_path.open('r') as f:
        data = json.load(f)

    with torch.no_grad():
        sal_map = sal_map.numpy()
        print("Original sal map shape = " + str(sal_map.shape))
        # 1. cut the saliency map into 5 segments
        # 1. Average over saliency map dimensions
        sal_map = np.expand_dims(sal_map, axis=1)[:,:,:-1,:] # remove 351st image
        sal_shape = sal_map.shape
        print("Sal shape = " + str(sal_shape))
        sal_map = np.reshape(sal_map, (sal_shape[0], sal_shape[2]//5, 5, sal_shape[3])) # should be 5x70x5x136

        # Average over each segment
        avg_sal_map = np.mean(sal_map, axis=1) # 5x5x136

        # 3. Convert targets into labels
        labels = []
        for elem in targets:
            label = data['database'][elem[0]]["annotations"]["label"]
            label = ACTION_DICT[label]
            labels.append(label)
        y = torch.LongTensor(labels).to(opt.device)
        print("y shape = " + str(y.shape))

        # For each video, find the 5 relevant images (img 0, 70, 140, 210, 280)
        relevant_indices = [i * (sal_shape[2]//5) for i in range(5)]
        image_name_formatter = lambda x: f'image_{x:05d}.jpg'
        image_names = [image_name_formatter(i) for i in relevant_indices]
        print("image names = " + str(image_names))

        img_root_dir = "/home/ubuntu/data/processed_video/binary_data_embed"
        #kpt_root_dir = "/home/ubuntu/data/processed_video/keypoints_binary_nose"
        
        # For each element in targets, get the relevant image paths and keypoint paths
        # For each relevant image, get the saliency map
        # Plot the image in the background, then scatter the keypoints using saliency as heatmap color
        for i in range(len(targets)):
            elem = targets[i]
            videoname = elem[0]
            classname = data['database'][videoname]["annotations"]["label"]
            img_dirpath = img_root_dir + "/" + classname + "/" + videoname
            #kpt_dirpath = kpt_root_dir + "/" + classname + "/" + videoname
            for j in range(len(image_names)):
                img_name = image_names[j]
                img_path = img_dirpath + "/" + img_name

                # Get keypoints
                img = cv2.imread(str(img_path))
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_img, 1)
                height, width = gray_img.shape
                face = dlib.rectangle(left=0, top=0, right=width-1, bottom=height-1)
                if (len(faces) > 0):
                    face = faces[0]
                keypoints = predictor(gray_img, face)
                keypoints = face_utils.shape_to_np(keypoints) # shape should be (68, 2)

                # Get saliency map
                sals = avg_sal_map[i][j] # numpy array of length 136
                sals = np.reshape(sals, (68, 2))
                sals = np.mean(sals, axis=1) # shape is (68,)

                plt.subplot(2, len(image_names), j+1)
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                im = plt.imread(img_path) #[top:bottom, left:right]
                #implot = plt.imshow(im)
                plt.scatter(keypoints[:,0], keypoints[:,1], s=5, c=sals, cmap=plt.cm.hot)
                plt.axis("off")

            figpath = Path('/home/ubuntu/data/processed_video/salmaps/bin_phq/map_' + classname + "_" + videoname)
            plt.savefig(figpath)
    return None

def compute_saliency_maps(model, opt):
    # Generate tiny data loader
    # Loop through it to generate saliency maps
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    #tiny_video_path = Path('/Users/ruta/stanford/pac/mentalhealth/cropped_phi_data/jpg') #Path('/home/ruta/teeny_data/nturgb/jpg')
    #tiny_annotation_path = Path('/Users/ruta/stanford/pac/mentalhealth/mh_01.json') #Path('/home/ruta/teeny_data/ntu_01.json')
    tiny_data, collate_fn = get_inference_data(opt.video_path,
                                               opt.annotation_path,
                                               opt.dataset,
                                               opt.input_type,
                                               opt.file_type,
                                               opt.inference_subset,
                                               None, #spatial_transform,
                                               None) #temporal_transform)

    tiny_loader = torch.utils.data.DataLoader(tiny_data,
                                               batch_size=opt.inference_batch_size,
                                               shuffle=False,
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=None,
                                               worker_init_fn=worker_init_fn,
                                               collate_fn=collate_fn)

    saliency_maps = []
    for i, (inputs, targets) in enumerate(tiny_loader):
        sal_map = get_saliency_map(inputs, targets, model, opt)
        # Plot the saliency map using matplotlib and save to a file
        plot_saliency_3d(sal_map, i, inputs, targets, opt)
        saliency_maps.append(sal_map)

    return saliency_maps

def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.distributed, opt.device)

    if opt.pretrain_path:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    else:
        parameters = model.parameters()

    if opt.is_master_node:
        print(model)

    label_path = "/home/ubuntu/data/processed_video/phq9_binary_labels/trainlist01.txt"
    if opt.mhq_data == "gad7":
        label_path = "/home/ubuntu/data/processed_video/gad7_binary_labels/trainlist01.txt"
    if opt.n_classes == 4:
        label_path = "/home/ubuntu/data/processed_video/phq9_multiclass_labels/trainlist01.txt"
        if opt.mhq_data == "gad7":
            label_path = "/home/ubuntu/data/processed_video/gad7_multiclass_labels/trainlist01.txt"
    if opt.label_path is not None:
        label_path = opt.label_path

    labels = []
    with open(label_path, 'r') as f:
        labels = torch.IntTensor([int(line.split(" ")[1]) for line in f])
    if (len(labels) == 0):
        print("LABELS IS EMPTY")
    weights = compute_class_weight(labels, opt.n_classes)

    #weights = [0.0 for i in range(opt.n_classes)]
    #for i in range(opt.n_classes):
    #    class_count = sum([1 if x==i else 0 for x in labels])
    #    weights[i] = len(labels) / class_count
    #weights = torch.FloatTensor(weights)

    print("weights = " + str(weights))
    #criterion = CrossEntropyLoss(weights).to(opt.device)
    criterion = CrossEntropyLoss().to(opt.device)
    # ADDED for 231n
    #criterion = FocalLoss(gamma=opt.fl_gamma).to(opt.device)

    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    conf_mtx_dict = {} # ADDED for CS231n

    prev_val_loss = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed, class_weights=weights)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer,
                                      opt.distributed,
                                      conf_mtx_dict, # ADDED for CS231n
                                      class_weights=weights)

        # ADDED for 231n - uncomment if using cross entropy loss
        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

    if opt.inference:
        inference_loader, inference_class_names = get_inference_utils(opt)
        inference_result_path = opt.result_path / '{}.json'.format(
            opt.inference_subset)

        inference.inference(inference_loader, model, inference_result_path,
                            inference_class_names, opt.inference_no_average,
                            opt.output_topk)

    # ADDED for CS231n
    compute_saliency_maps(model, opt)
    conf_mtx_file = csv.writer(open("conf_mtxs.csv", "w+"))
    for key, val in conf_mtx_dict.items():
        conf_mtx_file.writerow([key, val])

if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)
