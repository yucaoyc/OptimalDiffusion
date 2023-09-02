import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('./')
sys.path.append('../')
from lib.sdes import VariancePreservingSDE, PluginReverseSDE
from lib.flows.elemwise import LogitTransform
from lib.models.unet import UNet
from lib.helpers import logging, create
from tensorboardX import SummaryWriter
import json
from tqdm import tqdm
from torch.utils import collect_env
import yaml
from datetime import datetime
import time

def load_cifar_traindata_and_model(args):
    """
        Load CIFAR-10 trainloader and sde model
    """
    # LOAD CIFAR-10 MODEL
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), \
            transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args['dataroot'], \
            'cifar10'), train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                            shuffle=True, \
                                            num_workers=args['num_workers'])
    drift_q = UNet(
        input_channels=args['input_channels'],
        input_height=args['input_height'],
        ch=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
    )

    T = torch.nn.Parameter(torch.FloatTensor([args['T0']]), requires_grad=False)
    inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T)
    gen_sde = PluginReverseSDE(inf_sde, drift_q, T, \
            vtype=args['vtype'], debias=args['debias'])

    return trainloader, gen_sde

def load_weight(args):
    if args['weight'] == 'default':
        w=lambda z : torch.ones_like(z)
    elif args['weight'] == 'data':
        w=lambda z : torch.ones_like(z)/(0.25 + z)
    elif args['weight'] == 'noise':
        w=lambda z : z
    return w

def load_path_and_record_env(args, folder_path_postfix):
    #_folder_name_keys = ['dataset', 'weight', 'real', 'debias', \
    #        'batch_size', 'lr', 'num_iterations', 'T0', 'seed', 'num_method']
    _folder_name_keys = ['dataset', 'weight', 'batch_size', 'num_iterations', 'T0', 'seed', 'use_lightning', 'use_mixed_precision']
    folder_name = '-'.join([str(args[k]) for k in _folder_name_keys])
    create(args['saveroot'], folder_name)

    folder_path = os.path.join(args['saveroot'], folder_name)
    folder_path = folder_path + folder_path_postfix
    print_ = lambda s: logging(s, folder_path)
    print_(f'folder path: {folder_path}')
    with open(os.path.join(folder_path, 'args.txt'), 'w') as out:
        out.write(json.dumps(args, indent=4))
    with open(os.path.join(folder_path, 'env.txt'), 'w') as out:
        out.write(collect_env.get_pretty_env_info())
    writer = SummaryWriter(folder_path)
    return folder_path, writer, print_
