import os
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
sys.path.append('./')
sys.path.append('../')
from lib.helpers import create
from lib.fid import *
from lib.generate_batch import *
import matplotlib.pyplot as plt
import datetime
from cifar_util import *
import yaml
import warnings
import time
import pandas as pd

import click
@click.command()

#========================================
# required arguments
@click.option('--ckpt_path',            required=True,  type=str) # checkpoint path
@click.option('--alpha',                required=True,  type=float) # alpha = h/g

#========================================
# optional arguments
@click.option('--output_path',          default='',     type=str)
@click.option('--seed',                 default=0,      type=int) # for sampling only
@click.option('--device_id',            default=0,      type=int)
@click.option('--image_size',           default=32,     type=int)
@click.option('--num_sample',           default=10500,  type=int)
@click.option('--fid_batch_size',       default=750,    type=int)
@click.option('--num_steps',            default=1000,   type=int)
@click.option('--verbose',              default=False,  type=bool)
@click.option('--num_method',           default='ei',   type=click.Choice(['em', 'ei']))
# Remark: 750*14 = 10,500
@click.option('--use_fid_inception',    default=True,   type=bool)
@click.option('--save_img',             default=True,   type=bool)


def cifar_compute_fid(ckpt_path, alpha, \
        output_path, seed, device_id, \
        image_size, num_sample, fid_batch_size, num_steps, \
        verbose, num_method, use_fid_inception, save_img):

    if not os.path.exists(ckpt_path):
        warnings.warn('The checkpoint does not exists! Abort the calculations!')
        return

    if output_path == '':
        output_path = ckpt_path + '_{:d}_{:s}_data/'.format(num_steps, str(use_fid_inception))
    device = torch.device("cuda:"+str(device_id))

    #######################################
    # initialize the model
    with open('cifar.yaml', 'r') as file:
        args = yaml.safe_load(file)
    _, gen_sde = load_cifar_traindata_and_model(args)
    gen_sde.to(device)
    gen_sde.load_state_dict(torch.load(ckpt_path, map_location=device))
    logit = LogitTransform(alpha=args["logit_alpha"])

    # load v3 model
    v3model = load_v3model(use_fid_inception=use_fid_inception)
    v3model.to(device)

    #######################################
    # set seed for reproducibility.
    torch.manual_seed(seed)

    #######################################
    # compute FID
    trans_resize_toimg = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size)])
    img_to_tensor = transforms.ToTensor()

    # load existing pre-computed data
    pre_filename = "data/cifar_pretrained_stats_{:d}_{:s}.pt".format(image_size, str(use_fid_inception))
    if not os.path.exists(pre_filename):
        warnings.warn('pretrained_stats for cifar-10 has not been computed!')
        return
    else:
        trainset_mu, trainset_sigma = torch.load(pre_filename)

    # create data folder
    path_of_data = output_path
    create(path_of_data)

    # compute the statistics for generative model
    gen_mu, gen_sigma = generated_stats(gen_sde, v3model, device, \
            alpha, logit, args, num_sample, fid_batch_size, \
            trans_resize_toimg, image_size, \
            num_method = num_method, num_steps=num_steps, \
            path_of_data=path_of_data, verbose=verbose, save_img=save_img);

    # compute FID
    fid_value = calculate_frechet_distance(trainset_mu.numpy(), trainset_sigma.numpy(), \
            gen_mu, gen_sigma)
    print("FID for alpha={:.2f} is {:.2f}".format(alpha, fid_value))

    # save data to file.
    fid_data_fn = 'fid.pkl'
    if os.path.exists(fid_data_fn):
        fid_data_table = pd.read_pickle(fid_data_fn)
        fid_data_table = fid_data_table._append(pd.DataFrame({'checkpoint': ckpt_path, 'alpha': alpha, 'num_steps': num_steps, 'FID': fid_value}, index=[len(fid_data_table)]))
    else:
        fid_data_table = pd.DataFrame({'checkpoint': ckpt_path, 'alpha': alpha, 'num_steps': num_steps, 'FID': fid_value}, index=[0])
    fid_data_table.to_pickle(fid_data_fn)
    # to include later.

if __name__ == '__main__':
    cifar_compute_fid()
