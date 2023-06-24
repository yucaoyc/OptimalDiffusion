#!-*- coding:utf-8 -*-

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.special import kl_div

def add_noise(density: np.ndarray, eps: float=1e-7):

    total = np.sum(density)
    density = density + eps
    density = density * total / np.sum(density)

    return density

def evaluate_1d(true_data: np.ndarray, fake_data: np.ndarray, x0: float, x1: float):

    bins = np.linspace(start=x0, stop=x1, num=100)

    true_density, _ = np.histogram(a=true_data, bins=bins, range=(x0, x1), density=True)
    fake_density, _ = np.histogram(a=fake_data, bins=bins, range=(x0, x1), density=True)

    js = jensenshannon(p=true_density, q=fake_density)
    kl = kl_div(add_noise(true_density, eps=1e-7), add_noise(fake_density, eps=1e-7)).mean()
    wd = wasserstein_distance(u_values=true_density, v_values=fake_density)

    return js, kl, wd

def evaluate_2d(true_data: np.ndarray, fake_data: np.ndarray, x0: float, x1: float, y0: float, y1: float):

    xbins = np.linspace(start=x0, stop=x1, num=101)
    ybins = np.linspace(start=y0, stop=y1, num=101)

    true_density, *_ = np.histogram2d(x=true_data[:, 0], y=true_data[:, 1], bins=[xbins, ybins], range=[[x0, x1], [y0, y1]], density=True)
    fake_density, *_ = np.histogram2d(x=fake_data[:, 0], y=fake_data[:, 1], bins=[xbins, ybins], range=[[x0, x1], [y0, y1]], density=True)

    true_density = true_density.reshape(-1)
    fake_density = fake_density.reshape(-1)

    js = jensenshannon(p=true_density, q=fake_density)
    kl = kl_div(add_noise(true_density, eps=1e-15), add_noise(fake_density, eps=1e-15)).mean()
    wd = wasserstein_distance(u_values=true_density, v_values=fake_density)

    return js, kl, wd
