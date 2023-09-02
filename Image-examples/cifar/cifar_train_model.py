import os
import numpy as np
import torch
import torchvision
import sys
sys.path.append('./')
sys.path.append('../')
from lib.plotting import get_grid, plot_grids
from lib.flows.elemwise import LogitTransform
from lib.helpers import logging, create
from lib.fid import *
from tensorboardX import SummaryWriter
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils import collect_env
from cifar_util import *
import yaml
from datetime import datetime
import time
import lightning as L
from termcolor import colored

import click
@click.command()

@click.option('--weight_type',          required=True,       type=click.Choice(['default','data','noise']))
@click.option('--seed',                 required=True,       type=int)
@click.option('--num_iterations',       required=True,       type=int)
@click.option('--num_method',           required=True,       type=click.Choice(['em', 'ei']))

# optional parts
@click.option('--use_lightning',        default=True,        type=bool)
@click.option('--retrain_model',        default=False,       type=bool)
@click.option('--to_plot',              default=False,       type=bool)
@click.option('--quiet',                default=False,       type=bool)
@click.option('--use_mixed_precision',  default=True,        type=bool)

# we can either resume a training or enforce training using a particular checkpoint.
@click.option('--resume_train',         default=True,        type=bool)
@click.option('--ckpt_path',            default="",          type=str)
@click.option('--fabric_device_id',     default=[0],         type=list)
@click.option('--non_fabric_device_id', default=0,           type=int)
@click.option('--seed_training',        default=-1,          type=int)
@click.option('--save_ckpt_every_iter', default=True,        type=bool)
@click.option('--folder_path_postfix',  default="",          type=str)

def train_model(weight_type, seed, num_iterations, num_method, \
        use_lightning, retrain_model, to_plot, quiet, use_mixed_precision, \
        resume_train, ckpt_path, fabric_device_id, non_fabric_device_id, seed_training, save_ckpt_every_iter, folder_path_postfix):

    print("use mixed = {:s}".format(str(use_mixed_precision)))

    fabric_device_id = list(map(int,fabric_device_id)) # convert to integers
    if seed_training < 0:
        seed_training = seed

    #========================================
    # setup: fabric or not
    if use_lightning:
        torch.set_float32_matmul_precision('high')
        if use_mixed_precision:
            fabric = L.Fabric(accelerator="cuda", devices=fabric_device_id, \
                    strategy="auto", precision='16-mixed')
        else:
            fabric = L.Fabric(accelerator="cuda", devices=fabric_device_id, \
                    strategy="auto")
        fabric.launch()
        device = fabric.device
    else:
        device = torch.device("cuda:{:d}".format(non_fabric_device_id))
    torch.cuda.empty_cache()

    #========================================
    # load arguments from cifar.yaml
    with open('cifar.yaml', 'r') as file:
        args = yaml.safe_load(file)
    particular_str = "{:s}-{:s}-{:d}-{:d}".format(args['dataset'], \
            weight_type, seed, num_iterations)
    args['weight'] = weight_type
    args['seed'] = seed
    args['num_iterations'] = num_iterations
    args['num_method'] = num_method
    args['use_lightning'] = use_lightning
    args['use_mixed_precision'] = use_mixed_precision

    #========================================
    # load weight function
    w = load_weight(args)

    #========================================
    # folder and environment writter
    folder_path, writer, print_ = load_path_and_record_env(args, folder_path_postfix)

    #========================================
    # load trainloader and create a data type for gen_sde.
    torch.manual_seed(seed)
    # initialize model
    trainloader, gen_sde = load_cifar_traindata_and_model(args)
    gen_sde.to(device)
    total_params = sum(p.numel() for p in gen_sde.parameters())
    total_params_trainable = sum(p.numel() for p in gen_sde.parameters() if p.requires_grad)
    print(colored("Total # of params = {:d}".format(total_params),'blue')) if not quiet else print("")
    print(colored("Total # of trainable params = {:d}".format(total_params_trainable),'blue')) if not quiet else print("")

    #========================================
    # setup optimizer and logit transform
    optim = torch.optim.Adam(gen_sde.parameters(), lr=args['lr'])

    logit = LogitTransform(alpha=args["logit_alpha"])
    reverse = logit.reverse if args['real'] else None

    #========================================
    # decide what to do
    # * If retrain_model==True, we always train model from scratch.
    # * Otherwise,
    #   - if we allow resume training, we try to load the existing checkpoint
    #   - if we don't allow resume training, we can try to load a specified checkpoint

    # by default,
    not_finished, count, loss_vec = True, 0, np.zeros(args['num_iterations'])

    # Load model
    if not retrain_model:
        if ckpt_path != "":
            gen_sde.load_state_dict(torch.load(ckpt_path, map_location=device))

        # if a checkpoint exists, we simply load it.
        if os.path.exists(os.path.join(folder_path, 'checkpoint.pt')):
            gen_sde.load_state_dict(torch.load(os.path.join(folder_path, 'checkpoint.pt'), \
                    map_location=device))
            optim.load_state_dict(torch.load(os.path.join(folder_path, 'optim.pt')))
            not_finished, count, loss_vec = torch.load(os.path.join(folder_path, "status.pt"))

            if not_finished == False:
                print(colored("The model is fully trained and we simply load the checkpoint!",'blue'))
                return
            else:
                print(colored("The model does not exists! We shall restart training!",'blue'))
    # end of decision

    writer.add_scalar('T', gen_sde.T.item(), count)

    # set up fabric
    if use_lightning:
        gen_sde, optim = fabric.setup(gen_sde, optim)
        trainloader = fabric.setup_dataloaders(trainloader)

    #========================================
    # we start the training
    torch.manual_seed(seed_training)
    start = time.time()

    while not_finished:
        # turn on the training mode
        gen_sde.train()

        for x, _ in trainloader:
            # for each batch x has size [64, 1, 28, 28]
            if not use_lightning:
                x = x.to(device)

            # regularize the data
            x = x * 255 / 256 + torch.rand_like(x) / 256
            if args['real']:
                x, _ = logit.forward_transform(x, 0)

            # these two lines are equivalent
            #loss = gen_sde.dsm(x, w=w).mean()
            loss = gen_sde(x, w=w).mean()

            optim.zero_grad()
            if use_lightning:
                fabric.backward(loss)
            else:
                loss.backward()
            optim.step()

            loss_vec[count] = loss.item()

            count += 1
            if count % args['print_every'] == 0:
                writer.add_scalar('loss', loss.item(), count)
                writer.add_scalar('T', gen_sde.T.item(), count)
                print("{:s} - Iteration={:8d} Avg Loss={:8.2E} cuda={:1d} params={:s}".format(datetime.now().isoformat(timespec='seconds'), \
                        count, np.mean(loss_vec[(count-args['print_every']):count]), device.index, particular_str))

            if count >= args['num_iterations']:
                not_finished = False
                print_('Finished training')
                break

            if count % args['checkpoint_every'] == 0:
                torch.save(gen_sde.state_dict(), os.path.join(folder_path,'checkpoint.pt'))
                torch.save(optim.state_dict(), os.path.join(folder_path,'optim.pt'))
                torch.save([not_finished, count, loss_vec], os.path.join(folder_path, 'status.pt'))
                if save_ckpt_every_iter:
                    torch.save(gen_sde.state_dict(), os.path.join(folder_path,'checkpoint-iter-{:d}.pt'.format(count)))

            torch.cuda.empty_cache()

    # end of training
    torch.save(gen_sde.state_dict(), os.path.join(folder_path,'checkpoint.pt'))
    torch.save(optim.state_dict(), os.path.join(folder_path,'optim.pt'))
    torch.save([not_finished, count, loss_vec], os.path.join(folder_path, 'status.pt'))

    end = time.time()
    elapsed = end-start
    print("Time elapsed {:.2f} mins".format(elapsed/60))
    print("Memory used: {:.2f} GB".format(torch.cuda.max_memory_reserved()/1.0e9))

    # turn off the training mode
    gen_sde.eval()

    if to_plot:

        plt.figure(figsize=(4,3))
        plt.plot(np.arange(0, stop=args['num_iterations']), loss_vec)
        plt.title("Loss")
        plt.yscale("log")
        plt.savefig(folder_path + "/loss.eps")
        plt.savefig(folder_path + "/loss.pdf")

        # plot 5x5 figures
        print("start to generate samples")
        torch.manual_seed(seed);
        create(folder_path, "assets")
        num_steps = 1000

        for c in tqdm([0.0, 0.5, 1.0, 1.5, 2.0]):
            torch.manual_seed(seed)
            fig, gen_img, gen_img_data = plot_grids(gen_sde, c, logit, args, \
                                                    num_method=num_method, num_steps=num_steps, \
                                                    n=5, s=8, binary=False)
            plt.savefig(folder_path + "/assets/img_{:s}_{:.1f}.eps".format(particular_str,c))
            plt.savefig(folder_path + "/assets/img_{:s}_{:.1f}.pdf".format(particular_str,c))
            torch.save((gen_img, gen_img_data), \
                    folder_path + "/assets/{:s}_img_data_{:.1f}.pt".format(args['dataset'],c))

if __name__ == "__main__":
    train_model()
