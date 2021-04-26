import torch
from torch import nn

from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet

from embednet import EmbedNet


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'crnn', 'embednet'
    ]

    if opt.model == 'resnet' or opt.model == 'crnn':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=opt.model_depth,
                                          n_classes=opt.n_classes,
                                          n_input_channels=opt.n_input_channels,
                                          shortcut_type=opt.resnet_shortcut,
                                          conv1_t_size=opt.conv1_t_size,
                                          conv1_t_stride=opt.conv1_t_stride,
                                          no_max_pool=opt.no_max_pool,
                                          widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'wideresnet':
        model = wide_resnet.generate_model(
            model_depth=opt.model_depth,
            k=opt.wide_resnet_k,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'resnext':
        model = resnext.generate_model(model_depth=opt.model_depth,
                                       cardinality=opt.resnext_cardinality,
                                       n_classes=opt.n_classes,
                                       n_input_channels=opt.n_input_channels,
                                       shortcut_type=opt.resnet_shortcut,
                                       conv1_t_size=opt.conv1_t_size,
                                       conv1_t_stride=opt.conv1_t_stride,
                                       no_max_pool=opt.no_max_pool)
    elif opt.model == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'densenet':
        model = densenet.generate_model(model_depth=opt.model_depth,
                                        n_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)

    # Add layers if it is a crnn
    if opt.model == 'crnn':
        n_finetune_classes = opt.n_classes
        fc_hidden1, fc_hidden2, fc_hidden3 = 1024, 512, 256
        #modules = list(model.children())[:-1] # delete the last fc layer.
        model.fc = nn.Linear(model.fc.in_features, fc_hidden1)
        #tmp_model.fl1 = nn.Flatten(fc_hidden1, -1)
        model.end_bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        model.end_fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        #tmp_model.fl2 = nn.Flatten(fc_hidden2, -1)
        model.end_bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        model.end_fc3 = nn.Linear(fc_hidden2, fc_hidden3)
        model.end_bn3 = nn.BatchNorm1d(fc_hidden3, momentum=0.01)
        model.end_fc4 = nn.Linear(fc_hidden3, n_finetune_classes)

    if opt.model == 'embednet':
        model = EmbedNet(input_channels=1, 
                         n_channels=[1, 1, 1, 1, 1], 
                         kernel_size=5, 
                         dropout=0.5, 
                         lstm_n_hidden=256, 
                         lstm_n_layers=4, 
                         lstm_bidirectional=False, 
                         n_classes=opt.n_classes)

    return model


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)

        if model_name == 'crnn':
            fc_hidden1, fc_hidden2 = 1024, 512
            #modules = list(model.children())[:-1] # delete the last fc layer.
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features, fc_hidden1)
            #tmp_model.fl1 = nn.Flatten(fc_hidden1, -1)
            tmp_model.end_bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
            tmp_model.end_fc2 = nn.Linear(fc_hidden1, fc_hidden2)
            #tmp_model.fl2 = nn.Flatten(fc_hidden2, -1)
            tmp_model.end_bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
            tmp_model.end_fc3 = nn.Linear(fc_hidden2, n_finetune_classes)

    #return model
    return tmp_model


def make_data_parallel(model, is_distributed, device):
    print("Device type = " + str(device.type))
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model
