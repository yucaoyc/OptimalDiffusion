import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('./')
from lib.plotting import get_grid, plot_grids
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
from mnist_load_model import *
import time
import matplotlib.pyplot as plt

num_iterations = 2*10**4
_, args, logit, _ = load_model("default", 1, num_iterations, num_method="ei");

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root=args['dataroot'], train=True,
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST(root=args['dataroot'], train=False,
                                     download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                          shuffle=True, num_workers=args['num_workers'])
testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                         shuffle=True, num_workers=args['num_workers'])

Nt = 50
t_vec = np.linspace(1/Nt, args['T0'], Nt, endpoint=True)
weight_type_list = ["default", "data", "noise"]
option_list = ['k--', 'r--', 'b--']

# case 1
l_vec = np.zeros((3, Nt))
seed = 1
for weight_idx in range(3):
    weight_type = weight_type_list[weight_idx]
    gen_sde, _, _, _ = load_model(weight_type, seed, num_iterations, num_method="ei")
    for i in range(Nt):
        t_begin = time.time()
        l_vec[weight_idx, i] = gen_sde.dsm_at_time_t(t_vec[i], logit, testloader)
        print(time.time() - t_begin)
f1 = plt.figure(figsize=(7,2.5))
plt.subplot(1,2,1)
for weight_idx in range(3):
    plt.plot(t_vec/args['T0'], l_vec[weight_idx,:], option_list[weight_idx], label=weight_type_list[weight_idx])
plt.yscale("log")
plt.xlabel("t/T", fontsize=16)
plt.title("Loss in log scale", fontsize=16)
plt.subplot(1,2,2)
for weight_idx in range(3):
    plt.plot(t_vec/args['T0'], l_vec[weight_idx,:], option_list[weight_idx], label=weight_type_list[weight_idx])
plt.title("Loss in normal scale", fontsize=16)
plt.xlabel("t/T", fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
f1.savefig("../assets/loss_{:s}_seed_{:d}.eps".format(args['dataset'], seed))

# case 2
l_vec2 = np.zeros((3, Nt))
seed = 2
for weight_idx in range(3):
    weight_type = weight_type_list[weight_idx]
    gen_sde, _, _, _ = load_model(weight_type, seed, num_iterations, num_method="ei")
    for i in range(Nt):
        t_begin = time.time()
        l_vec2[weight_idx, i] = gen_sde.dsm_at_time_t(t_vec[i], logit, testloader)
        print(time.time() - t_begin)
f2 = plt.figure(figsize=(7,2.5))
plt.subplot(1,2,1)
for weight_idx in range(3):
    plt.plot(t_vec/args['T0'], l_vec2[weight_idx,:], option_list[weight_idx], label=weight_type_list[weight_idx])
plt.yscale("log")
plt.xlabel("t/T", fontsize=16)
plt.title("Loss in log scale", fontsize=16)
plt.subplot(1,2,2)
for weight_idx in range(3):
    plt.plot(t_vec/args['T0'], l_vec2[weight_idx,:], option_list[weight_idx], label=weight_type_list[weight_idx])
plt.title("Loss in normal scale", fontsize=16)
plt.xlabel("t/T", fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
f2.savefig("../assets/loss_{:s}_seed_{:d}.eps".format(args['dataset'], seed))

# Relative error
f3=plt.figure(figsize=(5,4))
plt.plot(t_vec/args['T0'], l_vec[1,:]/l_vec[0,:], 'r-', label=weight_type_list[1] + ", trial 1")
plt.plot(t_vec/args['T0'], l_vec[2,:]/l_vec[0,:], 'b-', label=weight_type_list[2] + ", trial 1")
plt.plot(t_vec/args['T0'], l_vec2[1,:]/l_vec2[0,:], 'r-.', label=weight_type_list[1] + ", trial 2")
plt.plot(t_vec/args['T0'], l_vec2[2,:]/l_vec2[0,:], 'b-.', label=weight_type_list[2] + ", trial 2")
plt.plot(t_vec/args['T0'], np.ones(np.size(t_vec)), label="reference")
plt.xlabel("t/T", fontsize=16)
plt.legend(fontsize=14)
plt.title("Relative score-matching loss", fontsize=16)
plt.tight_layout()
f3.savefig("../assets/rela_loss_{:s}.eps".format(args['dataset']))

print(np.sum(l_vec, axis=1)/Nt)
print(np.sum(l_vec2, axis=1)/Nt)

with open("data/{:s}_l_vec.npy".format(args['dataset']), 'wb') as f:
    np.save(f, l_vec)
    np.save(f, l_vec2)
