import os
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
sys.path.append('./')
sys.path.append('../')
from lib.helpers import create
from lib.fid import *
import matplotlib.pyplot as plt
import datetime
from cifar_util import *
import yaml
import warnings
import time
import pandas as pd
import pickle
import click
@click.command()

#========================================
# required arguments
@click.option('--ckpt_path',            required=True,  type=str) # checkpoint path
@click.option('--nt',                   required=True,  type=int) # time_grids

#========================================
# optional arguments
@click.option('--seed',                 default=0,      type=int) # for sampling only
@click.option('--device_id',            default=0,      type=int)
@click.option('--verbose',              default=False,  type=bool)

def cifar_compute_sml(ckpt_path, nt, seed, device_id, verbose):

    if not os.path.exists(ckpt_path):
        warnings.warn('The checkpoint does not exists! Abort the calculations!')
        return

    device = torch.device("cuda:"+str(device_id))

    #######################################
    # initialize the model
    with open('cifar.yaml', 'r') as file:
        args = yaml.safe_load(file)
    _, gen_sde = load_cifar_traindata_and_model(args)
    gen_sde.to(device)
    gen_sde.load_state_dict(torch.load(ckpt_path, map_location=device))
    logit = LogitTransform(alpha=args["logit_alpha"])

    #######################################
    # set seed for reproducibility.
    torch.manual_seed(seed)

    #######################################
    # compute sml
    output_fn = ckpt_path + '_{:d}_sml.pkl'.format(nt)
    if not os.path.exists(output_fn):
        t_vec = np.linspace(args['T0']/nt, args['T0'], nt, endpoint=True)
        l_vec_single = np.zeros(nt)
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), \
            transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root=os.path.join(args['dataroot'], \
            'cifar10'), train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                            shuffle=True, \
                                            num_workers=args['num_workers'])
        for i in range(nt):
            if i % 10 == 0:
                t_begin = time.time()

            l_vec_single[i] = gen_sde.dsm_at_time_t(t_vec[i], logit, testloader)
            
            if (i+1) % 10 == 0 and verbose:
                print("iter = {:d} time = {:.2f} sec".format(i,time.time() - t_begin))
            torch.cuda.empty_cache()

        sml = np.sum(l_vec_single)*args['T0']/nt
        dict_data = {'tvec' : t_vec, 'smlvec' : l_vec_single, 'sml' : sml}
        pickle.dump(dict_data, open(output_fn,'wb'))
    else:
        dict_data = pickle.load(open(output_fn,'rb'))

if __name__ == '__main__':
    cifar_compute_sml()
