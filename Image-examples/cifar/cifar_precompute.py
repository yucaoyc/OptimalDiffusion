import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('./')
sys.path.append('../')
from lib.fid import *
from lib.helpers import create
import yaml
import time

# Precompute statistics for CIFAR10
def precompute_stats_for_cifar10(trainloader, model, device, dims=2048, verbose=False):

    final_mu = np.zeros(dims)
    final_secm = np.zeros((dims,dims))
    existing_size = 0

    model.to(device)

    for x, _ in trainloader:
        count_x = x.size(0)
        x = x.to(device)
        x = x * 255 / 256 + torch.rand_like(x) / 256 # we had regularized data during training
        mu, _, secm = calculate_activation_statistics(x, model)
        final_mu = (final_mu * existing_size + count_x * mu)/(existing_size + count_x)
        final_secm = (final_secm * existing_size + count_x * secm)/(existing_size + count_x)
        existing_size = existing_size + count_x

        if verbose and (existing_size % (64*10) == 0):
            print("Already computed stats for {:7d} images".format(existing_size))

    final_sigma = final_secm - vvt(final_mu)
    final_sigma = final_sigma * existing_size/(existing_size-1) # correct bias
    return final_mu, final_sigma

if __name__=='__main__':
    image_size = int(sys.argv[1])
    device_id = int(sys.argv[2])
    use_fid_inception = bool(sys.argv[3])
    device = torch.device("cuda:"+str(device_id))

    with open('cifar.yaml', 'r') as file:
        args = yaml.safe_load(file)

    # resize the trainloader to image_size x image_size
    transform_img_size = transforms.Compose([transforms.Resize(image_size), \
                                transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    trainset_img_size = torchvision.datasets.CIFAR10(root=os.path.join(args['dataroot'], 'cifar10'), train=True, \
                                download=True, transform=transform_img_size)
    trainloader_img_size = torch.utils.data.DataLoader(trainset_img_size, batch_size=args['batch_size'], \
                                shuffle=True, num_workers=args['num_workers'])

    v3model = load_v3model(use_fid_inception=use_fid_inception)
    v3model.to(device)

    create("data")
    pre_filename = "data/cifar_pretrained_stats_{:d}_{:s}.pt".format(image_size, str(use_fid_inception))

    if not os.path.exists(pre_filename):
        # Compute data
        begin_time = time.time()
        trainset_mu, trainset_sigma = precompute_stats_for_cifar10(trainloader_img_size, v3model, device, verbose=True)
        print("Total time used {:8.2f} seconds".format(time.time() - begin_time))
        torch.save((torch.tensor(trainset_mu), torch.tensor(trainset_sigma)), pre_filename)
