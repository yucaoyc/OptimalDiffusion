import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('./')
sys.path.append('../')
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

retrain_model = False

def load_model(weight_type, seed, num_iterations, num_method="ei", to_plot=False, device=None):

    if device == None:
        # use default configuration
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    with open('mnist.yaml', 'r') as file:
        args = yaml.safe_load(file)
    particular_str = "{:s}-{:s}-{:d}-{:d}".format(args['dataset'], \
            weight_type,seed,num_iterations)

    args['weight'] = weight_type
    args['seed'] = seed
    args['num_iterations'] = num_iterations
    args['num_method'] = num_method

    if args['weight'] == 'default':
        w=lambda z : torch.ones_like(z)
    elif args['weight'] == 'data':
        w=lambda z : torch.ones_like(z)/(0.25 + z)
    elif args['weight'] == 'noise':
        w=lambda z : z

    # training
    torch.manual_seed(seed)
    num_workers = args['num_workers']
    input_channels = args['input_channels']
    input_height = args['input_height']
    dimx = input_channels * input_height ** 2

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=args['dataroot'], train=True,
                                          download=True, transform=transform)
#    testset = torchvision.datasets.MNIST(root=args['dataroot'], train=False,
#                                         download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                              shuffle=True, num_workers=args['num_workers'])
#    testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
#                                             shuffle=True, num_workers=args['num_workers'])

    drift_q = UNet(
        input_channels=input_channels,
        input_height=input_height,
        ch=32,
        #ch_mult=(1, 2, 2),
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
    print_ = lambda s: logging(s, folder_path)
    print_(f'folder path: {folder_path}')
    with open(os.path.join(folder_path, 'args.txt'), 'w') as out:
        out.write(json.dumps(args, indent=4))
    with open(os.path.join(folder_path, 'env.txt'), 'w') as out:
        out.write(collect_env.get_pretty_env_info())
    writer = SummaryWriter(folder_path)

    T = torch.nn.Parameter(torch.FloatTensor([args['T0']]), requires_grad=False)
    inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T)
    gen_sde = PluginReverseSDE(inf_sde, drift_q, T, \
            vtype=args['vtype'], debias=args['debias'])

    # load model to particular device
    inf_sde.to(device)
    gen_sde.to(device)

    total_params = sum(p.numel () for p in gen_sde.parameters ())
    print("Total number of parameters = {:d}".format(total_params))

    optim = torch.optim.Adam(gen_sde.parameters(), lr=args['lr'])

    logit = LogitTransform(alpha=args["logit_alpha"])
    if args['real']:
        reverse = logit.reverse
    else:
        reverse = None

    ###############################################################################
    # Training
    loss_vec = np.zeros(args['num_iterations'])
    torch.manual_seed(seed);
    if os.path.exists(os.path.join(folder_path, 'checkpoint.pt')) and (retrain_model==False):
        if device.index == None:
            device_str = device.type
        else:
            device_str = device.type+":"+str(device.index)
        gen_sde, optim, not_finished, count, loss_vec = torch.load(os.path.join(folder_path, 'checkpoint.pt'), map_location=device_str)
        if not_finished == False:
            print("The model is fully trained and we simply load the checkpoint!")
            return gen_sde, args, logit, folder_path
    else:
        not_finished = True
        count = 0
        writer.add_scalar('T', gen_sde.T.item(), count)

    while not_finished:
        # turn on the training mode
        gen_sde.train()

        for x, _ in trainloader:
            x = x.to(device)
            x = x * 255 / 256 + torch.rand_like(x) / 256
            if args['real']:
                x, _ = logit.forward_transform(x, 0)

            loss = gen_sde.dsm(x, w=w).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_vec[count] = loss.item()

            count += 1
            if count == 1 or count % args['print_every'] == 0:
                writer.add_scalar('loss', loss.item(), count)
                writer.add_scalar('T', gen_sde.T.item(), count)
                print("{:s}\t Iteration={:9d}\t Loss={:.4f}".format( \
                        str(datetime.datetime.now()), count, loss.item()))

            if count >= args['num_iterations']:
                not_finished = False
                print_('Finished training')
                break

            if count % args['checkpoint_every'] == 0:
                torch.save([gen_sde, optim, not_finished, count, loss_vec], \
                           os.path.join(folder_path, 'checkpoint.pt'))

            torch.cuda.empty_cache()

    if not_finished == False:
        torch.save([gen_sde, optim, not_finished, count, loss_vec], os.path.join(folder_path, 'checkpoint.pt'))

    if to_plot:
        plt.figure(figsize=(4,3))
        plt.plot(np.arange(0, stop=args['num_iterations']), loss_vec)
        plt.title("Loss")
        plt.yscale("log")
        plt.savefig(folder_path + "/loss_{:s}.eps".format(particular_str))
        plt.savefig(folder_path + "/loss_{:s}.pdf".format(particular_str))

        # turn off the training mode
        gen_sde.eval()

        # plot 10 * 10 figures
        print("start to generate samples")
        torch.manual_seed(seed);
        create(folder_path, "assets")
        num_steps = 200
        for c in tqdm(np.sqrt(np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))):
            torch.manual_seed(seed)
            fig, gen_img, gen_img_data = plot_grids(gen_sde, c, logit, args, num_method=num_method, \
                                                    num_steps=num_steps, n=10, s=8, binary=True)
            plt.savefig(folder_path + ("/assets/img_{:s}_{:.2f}".format(particular_str,c)).replace('.','-') + ".eps")
            plt.savefig(folder_path + ("/assets/img_{:s}_{:.2f}".format(particular_str,c)).replace('.','-') + ".pdf")
            torch.save((gen_img, gen_img_data), folder_path + "/assets/{:s}_img_data_{:.2f}.pt".format(args['dataset'],c))

    return gen_sde, args, logit, folder_path
