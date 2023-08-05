# Modification based on https://github.com/CW-Huang/sdeflow-light/blob/main/lib/plotting.py

import torch
import matplotlib.pyplot as plt

def get_grid(sde, input_channels, input_height, n=4, num_steps=20, transform=None,
        mean=0, std=1, clip=True, return_tensor_data=False,
        method="em", alpha=-1):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    dt = delta.item()
    with torch.no_grad():
        for i in range(num_steps):
            t = i * delta.item()
            if method == "em":
                y0 = sde.em(y0, ts[i].item(), dt, alpha)
            elif alpha >= 0:
                y0 = sde.ei(y0, t, dt, alpha)
            else:
                print("The numerical method is not yet implemented!")

    if return_tensor_data == False:
        y0data = None
    else:
        y0data = y0.clone().detach()

    if transform is not None:
        y0 = transform(y0)

    if clip:
        y0 = torch.clip(y0, min=0, max=1)

    y0 = y0.view(
        n, n, input_channels, input_height, input_height).permute(
        2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)

    # remark:
    # y0data in R^d space and y0 possibly in image space specified by transform

    return y0.cpu().numpy(), y0data

def plot_grids(gen_sde, c, logit, args, \
        num_method="ei", num_steps=None, n=10, s=8, binary=False):
    # use default value
    if num_steps == None:
        num_steps = args['num_steps']

    # set h = c g
    # gen_sde.h = lambda t, y, T : c * gen_sde.base_sde.g(t, y)
    # gen_sde.h_name = str(c)

    # transform is the mapping to transform R^d data into images.
    transform_to_img = lambda x : logit.reverse(x, 0)[0]

    gen_img, gen_img_data = get_grid(gen_sde, \
        args['input_channels'], args['input_height'], n = n,
        num_steps=num_steps, transform=transform_to_img,
        clip=True, return_tensor_data=True, method=num_method, alpha=c)

    fig = plt.figure(figsize=(s,s))
    if binary:
        plt.imshow(gen_img[0,:,:], cmap='binary')
    else:
        plt.imshow(gen_img.transpose(1,2,0))
    plt.tight_layout()
    plt.axis('off')
    return fig, gen_img, gen_img_data
