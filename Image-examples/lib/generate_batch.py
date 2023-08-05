import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

def generate_batch_sample(sde, alpha, logit, args, \
        num_method="ei", num_steps=None, num_samples = 10, clip=True, mean=0, std=1, \
        verbose=False):

    t_begin = time.time()

    # use default value
    if num_steps == None:
        num_steps = args['num_steps']

    # transform is the mapping to transform R^d data into images.
    transform = lambda x : logit.reverse(x, 0)[0]

    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, args['input_channels'], args['input_height'], args['input_height']).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    dt = delta.item()
    with torch.no_grad():
        for i in range(num_steps):
            t = i * delta.item()
            if num_method == "em":
                y0 = sde.em(y0, ts[i].item(), dt, alpha)
            elif alpha >= 0:
                y0 = sde.ei(y0, t, dt, alpha)
            else:
                print("The numerical method is not yet implemented!")

    y0 = transform(y0)
    if clip:
        y0 = torch.clip(y0, min=0, max=1)

    if verbose:
        print("Time spent on the generative process is {:.2f}".format(time.time() - t_begin))

    return y0

def save_batch_sample(y0, to_save_path, img_transform, \
        new_size = 64, start_idx=0, verbose=False, save_img=False):

    img_to_tensor = transforms.ToTensor()
    y0_new = torch.zeros(y0.size(0), y0.size(1), new_size, new_size).to(y0)

    t_begin = time.time()

    for i in range(y0.size(0)):
        img = img_transform(y0[i,:,:,:])
        if save_img:
            img.save(to_save_path+"/"+str(start_idx+i)+".png")
        y0_new[i,:,:,:] = img_to_tensor(img)

    if verbose:
        print("Time spent on the postprocessing is {:.2f}".format(time.time() - t_begin))

    return y0_new

def plot_generated_batch_samples(args, y0, n, s=8):
    if y0.size(0) < n**2:
        print("n is too large")
        return

    y0 = y0[0:(n**2),:,:,:]
    y0 = y0.view(
        n, n, args['input_channels'], args['input_height'], args['input_height']).permute(
        2, 0, 3, 1, 4).contiguous().view(args['input_channels'], n * args['input_height'], n * args['input_height'])

    fig = plt.figure(figsize=(s,s))
    plt.imshow(y0.cpu().numpy().transpose(1,2,0))
    plt.tight_layout()
    plt.axis('off')
