import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys
from lib.sdes import VariancePreservingSDE, PluginReverseSDE
from lib.plotting import get_grid, plot_grids
from lib.flows.elemwise import LogitTransform
from lib.models.unet import UNet
from lib.helpers import logging, create
from tensorboardX import SummaryWriter
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils import collect_env
import yaml
import datetime

def load_model(weight_type, seed, num_iterations, num_method):
    torch.cuda.empty_cache()
    with open('mnist.yaml', 'r') as file:
        args = yaml.safe_load(file)
    particular_str = "{:s}-{:s}-{:d}-{:d}".format(args['dataset'],weight_type,seed,num_iterations)

    args['weight'] = weight_type
    args['seed'] = seed
    args['num_iterations'] = num_iterations
    args['num_method'] = num_method

    # training
    torch.manual_seed(seed)
    num_workers = args['num_workers']
    input_channels = args['input_channels']
    input_height = args['input_height']
    dimx = input_channels * input_height ** 2

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=args['dataroot'], train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=args['dataroot'], train=False,
                                         download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                             shuffle=True, num_workers=num_workers)
    drift_q = UNet(
        input_channels=input_channels,
        input_height=input_height,
        ch=32,
        ch_mult=(1,2,),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
    )
    _folder_name_keys = ['dataset', 'weight', 'real', 'debias', \
                         'batch_size', 'lr', 'num_iterations', 'T0', 'seed', 'num_method']
    folder_name = '-'.join([str(args[k]) for k in _folder_name_keys])
    create(args['saveroot'], folder_name)

    folder_path = os.path.join(args['saveroot'], folder_name)
 
    T = torch.nn.Parameter(torch.FloatTensor([args['T0']]), requires_grad=False)
    inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T)
    gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype=args['vtype'], debias=args['debias'])

    if torch.cuda.is_available():
        gen_sde.cuda()

    logit = LogitTransform(alpha=args["logit_alpha"])

    if args['real']:
        reverse = logit.reverse
    else:
        reverse = None

    torch.manual_seed(seed);
 
    if os.path.exists(os.path.join(folder_path, 'checkpoint.pt')):
        gen_sde, _, _, _, _ = torch.load(os.path.join(folder_path, 'checkpoint.pt'))
        return gen_sde, logit, args
    else:
        print("File Not Found!")
        return None

#############################################
# Parameters and setup
weight_type_list = ['data','default','noise']
seed_list = [1, 2]
num_steps_list = [30, 60, 100, 200, 1000]
iter_num = 20000
num_method = 'ei'
c_vec = np.sqrt(np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
n = 10
data = torch.zeros(len(weight_type_list), len(seed_list), len(num_steps_list), len(c_vec), 28*n, 28*n)


#############################################
# get all data
for i in range(len(weight_type_list)):
    for j in range(len(seed_list)):
        for l in range(len(num_steps_list)):
            print((i,j,l))
            for c_idx in tqdm(range(len(c_vec))):
                weight_type = weight_type_list[i]
                seed = seed_list[j]
                num_steps = num_steps_list[l]
                c = c_vec[c_idx]
                torch.manual_seed(seed)

                gen_sde, logit, args = load_model(weight_type, seed, iter_num, num_method)
                # turn on evaluation mode
                gen_sde.eval()

                _, gen_img, _ = plot_grids(gen_sde, c, logit, args, num_method=num_method, \
                                            num_steps=num_steps, n=n, binary=True)
                data[i,j,l,c_idx,:,:] = torch.tensor(gen_img[0,:,:])
                plt.close()
# save data
torch.save(data, "img_data.pt")

#############################################
# Effect of weight
c_idx = 0 # ODE
num_step_idx = 4

print("{:2.1f}".format(c_vec[c_idx]**2))
print(num_steps_list[num_step_idx])

for seed_idx in [0, 1]:
    plt.figure(figsize=(9,3))
    for i in range(len(weight_type_list)):
        plt.subplot(1,3,i+1)
        plt.imshow(data[i,seed_idx,num_step_idx, c_idx,:,:], cmap='gray')
        plt.title('weight type = ' + weight_type_list[i], fontsize=14)
        plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig("../assets/MNIST_ODE_with_different_training" \
                + "{:d}-{:d}".format(seed_list[seed_idx], num_steps_list[num_step_idx]) \
                + ".eps")

#############################################
# Effect of weight & discretization steps
for weight_type_idx in [0, 1, 2]: #'default'
    for seed_idx in [0, 1]:
        num_row = 5
        num_col = 5
        plt.figure(figsize=(10,10))
        for k in range(num_row): # num step
            for i in range(num_col): # c
                plt.subplot(num_row, num_col, k*num_col + i + 1)
                plt.imshow(data[weight_type_idx, seed_idx, k, i, :, :], cmap='gray')
                if k == 0:
                    plt.title(r'$h^2/g^2 = {:2.1f}$'.format(c_vec[i]**2), fontsize=14)
                if i == 0:
                    plt.ylabel("step={:d}".format(num_steps_list[k]), fontsize=14)
                plt.xticks([]); plt.yticks([])
        plt.suptitle("weight type={:s}, trial={:d}".format(weight_type_list[weight_type_idx], \
                                                          seed_list[seed_idx]), fontsize=16)
        plt.tight_layout()
        plt.savefig("../assets/MNIST_effect_of_c_" + \
                    "{:s}_{:d}".format(weight_type_list[weight_type_idx], \
                    seed_list[seed_idx]) + ".eps")
