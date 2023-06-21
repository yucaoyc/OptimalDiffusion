#!-*- coding:utf-8 -*-

import numpy as np
from sklearn.datasets import make_swiss_roll


class SwissRoll(object):

    def __init__(self, noise: float=0.0, a = 0.2) -> None:
        super().__init__()

        self.noise = noise
        self.a = a

        self.data_dim = 2

        data, _ = make_swiss_roll(n_samples=10000, noise=self.noise)

        data = np.stack([data[:, 0], data[:, 2]], axis=1)

        self.x_max, self.x_min = data[:, 0].max(), data[:, 0].min()
        self.y_max, self.y_min = data[:, 1].max(), data[:, 1].min()

    def sample(self, n):

        data, _ = make_swiss_roll(n_samples=n, noise=self.noise)

        data = np.stack([data[:, 0], data[:, 2]], axis=1)

        data[:, 0] = 2 * self.a * (data[:, 0] - self.x_min) / (self.x_max - self.x_min) - self.a
        data[:, 1] = 2 * self.a * (data[:, 1] - self.y_min) / (self.y_max - self.y_min) - self.a

        return data

    def data_iter(self, batch_size: int=32, maxiter: int=1000):

        for i in range(maxiter):

            data = self.sample(batch_size)

            yield data

        return
