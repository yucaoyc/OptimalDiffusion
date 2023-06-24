#!-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def visualize_line(data: np.ndarray, xaxis=None, yscale=None, xl=None, yl=None, title=None, savename="line.png"):

    n = data.shape[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.set_tight_layout(True)

    if xaxis is None:
        xaxis = np.arange(n)

    ax.plot(xaxis, data)

    if yscale is not None:
        ax.set_yscale(yscale)

    if title is not None:
        ax.set_title(title, fontsize=16)

    if xl is not None:
        ax.set_xlabel(xl, fontsize=14)

    if yl is not None:
        ax.set_ylabel(yl, fontsize=14)

    plt.show()

    if savename is not None:
        fig.savefig(savename, dpi=300)
        
    plt.clf(); plt.cla(); plt.close()

def visualize_diffusion_process_1d(xs, titles, density_func=None, density_func_params={}, savename="diffusion_process.png", title=None):

    n = len(titles)

    fig, ax = plt.subplots(ncols=n, figsize=(n * 4, 4))
    fig.set_tight_layout(True)

    for i in range(n):

        try:
            _, bins, _ = ax[i].hist(xs[i], bins=100, density=True)

            if density_func is not None:

                x = np.linspace(np.min(bins), np.max(bins), num=201)[:, None]
                p = density_func(x, **density_func_params)
                ax[i].plot(x, p, "r", linewidth=2)

            ax[i].set_title(titles[i], fontsize=24)

        except TypeError:
            
            _, bins, _ = ax.hist(xs[i], bins=100, density=True)

            if density_func is not None:

                x = np.linspace(np.min(bins), np.max(bins), num=201)[:, None]
                p = density_func(x, **density_func_params)
                ax.plot(x, p, "r", linewidth=2)

    if title is not None:
        fig.suptitle(title, fontsize=20)

    plt.show()

    if savename is not None:
        fig.savefig(savename)

    plt.clf(); plt.cla(); plt.close()

def visualize_diffusion_process_2d(xs, titles, savename="diffusion_process.png"):

    n = len(titles)
    assert n == xs.shape[0]

    fig, ax = plt.subplots(ncols=n, figsize=(n * 4, 4))
    fig.set_tight_layout(True)

    for i in range(n):

        try:
            ax[i].scatter(xs[i, :, 0], xs[i, :, 1], alpha=0.5)
            ax[i].set_title(titles[i], fontsize=24)

        except:
            ax.scatter(xs[i, :, 0], xs[i, :, 1], alpha=0.5)
            ax.set_title(titles[i], fontsize=24)

    plt.show()

    if savename is not None:
        fig.savefig(savename)
        print(f"Image has been saved to {savename}")

    plt.clf(); plt.cla(); plt.close()


def visualize_diffusion_process_2d_marginal(x_t, titles, density_func=None, title=None, savename="diffusion_process.png"):

    n = x_t.shape[0]

    fig, ax = plt.subplots(ncols=n, nrows=2, figsize=(n * 4, 6))
    fig.set_tight_layout(True)

    for i in range(n):

        for d in range(2):

            try:
                ax_ = ax[d][i]

            except:
                ax_ = ax[d]
            
            _, bins, _ = ax_.hist(x_t[i, :, d], density=True, bins=100)
            ax_.set_title(titles[i], fontsize=24)

            if i == 0:
                ax_.set_ylabel(f"x_{d:d}", fontsize=24)

            if density_func is not None:

                x = np.linspace(np.min(bins), np.max(bins), num=201)[:, None]
                p = density_func(x, marg_dim = d)
                ax_.plot(x, p, "r", linewidth=4)
                ax_.tick_params(labelsize=20)

    if title is not None:
        fig.suptitle(title, fontsize=28)

    plt.show()
    if savename is not None:
        fig.savefig(savename, dpi=300)

    plt.clf(); plt.cla(); plt.close()
