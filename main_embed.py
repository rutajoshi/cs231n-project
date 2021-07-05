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
from focalloss import FocalLoss, compute_class_weight

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

    train_data = get_training_data(opt.video_path, opt.annotation_path,
                                   opt.dataset, opt.input_type, opt.file_type,
                                   None, temporal_transform)

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

    #val_data, collate_fn = get_validation_data(opt.video_path,
    #                                           opt.annotation_path, opt.dataset,
    #                                           opt.input_type, opt.file_type,
    #                                           spatial_transform,
    #                                           temporal_transform)
    
    val_data, collate_fn = get_validation_data(opt.video_path,
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,
                                               None,
                                               None)
    
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
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

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

    #inference_data, collate_fn = get_inference_data(
    #    opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
    #    opt.file_type, opt.inference_subset, spatial_transform,
    #    temporal_transform)

    inference_data, collate_fn = get_inference_data(
        opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
        opt.file_type, opt.inference_subset, None,
        None)
    
    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

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

def compute_saliency_maps(X, y, model):
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
    X.requires_grad()
    saliency = None

    #scores = model(X).gather(1, y.view(-1, 1)).squeeze().sum()
    scores = model(X)
    score_max_index = scores.argmax()
    score_max = scores[0,score_max_index]

    #scores.backward()
    score_max.backward()
    
    saliency, temp = X.grad.data.abs().max(dim = 1)

    return saliency

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
    criterion = CrossEntropyLoss(weights).to(opt.device)
    #criterion = CrossEntropyLoss().to(opt.device)
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
                        train_batch_logger, tb_writer, opt.distributed)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer,
                                      opt.distributed, 
                                      conf_mtx_dict) # ADDED for CS231n

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
