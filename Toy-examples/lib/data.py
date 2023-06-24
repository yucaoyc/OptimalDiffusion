#!-*- coding:utf-8 -*-

import numpy as np
from sklearn.datasets import make_swiss_roll

class GMM(object):

    def __init__(self, mus: np.ndarray=None, sigmas: np.ndarray=None, ps: np.ndarray=None, data_dim: int=1) -> None:
        super().__init__()

        self.data_dim = data_dim

        if mus is None:
            self.mus = np.zeros(self.data_dim,)

        else:
            assert mus.shape[1] == self.data_dim, f"means data_dim[1] {mus.shape[1]:d} != required data_dim {self.data_dim:d}"
            self.mus = mus

        if sigmas is None:
            self.sigmas = np.ones(1,)
        else:
            assert sigmas.shape[1] == 1, f"sigmas data_dim[1] {sigmas.shape[1]:d} != required 1"
            self.sigmas = sigmas

        if ps is None:
            self.ps = np.ones(1,)
        else:
            self.ps = ps

        self.modes = self.mus.shape[0]

        if not np.allclose(np.sum(self.ps), 1.0):
            raise ValueError

    def sample(self, n):

        selected_center_ids = np.random.choice(self.modes, n, p=self.ps)
        eps = np.random.normal(loc=0.0, scale=1.0, size=(n, self.data_dim,))
        data = eps * self.sigmas[selected_center_ids] + self.mus[selected_center_ids]

        return data

    def data_iter(self, batch_size: int=32, maxiter: int=1000):

        for i in range(maxiter):

            data = self.sample(batch_size)

            yield data

        return

    def p_t(self, x: np.ndarray, Bt: np.ndarray = None):

        size, dim = x.shape

        if Bt is None:
            Bt = np.zeros((size, 1))

        p = np.zeros((size, 1))
        a = np.exp(-Bt)

        for n in range(self.modes):
        
            mu_n = np.matmul(np.sqrt(a), self.mus[n, :].reshape(1, dim))
            var_n = a * (np.square(self.sigmas[n, 0]) - 1) + 1

            p_n = np.exp(- np.sum(np.square(x - mu_n) / var_n, axis=1, keepdims=True) / 2) / np.power(2 * np.pi * var_n, dim / 2)

            p += self.ps[n] * p_n

        return p

    def sample_t(self, Bt: float, n: int):

        selected_center_ids = np.random.choice(self.modes, n, p=self.ps)
        eps = np.random.normal(loc=0.0, scale=1.0, size=(n, self.data_dim,))

        a = np.exp(-Bt)

        m = np.sqrt(a) * self.mus
        v = a * (np.square(self.sigmas) - 1) + 1

        data = eps * np.sqrt(v)[selected_center_ids] + m[selected_center_ids]

        return data

    def exact_score_t(self, x: np.ndarray, Bt: np.ndarray):

        size, dim = x.shape

        a = np.exp(-Bt).reshape(-1, 1)
        p = np.zeros((size, 1))
        score_n = np.zeros_like(x)

        for n in range(self.modes):
        
            mu_n = np.sqrt(a) * self.mus[n, :].reshape(1, dim)
            var_n = a * (np.square(self.sigmas[n, 0]) - 1) + 1

            p_n = np.exp(- np.sum(np.square(x - mu_n) / var_n, axis=1, keepdims=True) / 2) / np.power(2 * np.pi * var_n, dim / 2)

            p += self.ps[n] * p_n
            score_n -= self.ps[n] * p_n * (x - mu_n) / var_n

        score = score_n / p

        return score

    def marginal(self, x: np.ndarray, marg_dim: int=0):

        m, s = self.mus[:, marg_dim], self.sigmas[:, 0]
        v = np.square(s).reshape(1, -1)

        p_m = 1 / np.sqrt(2 * np.pi * v) * np.exp(- np.square(x.reshape(-1, 1) - m.reshape(1, -1)) / v / 2)
        p = np.matmul(p_m, self.ps)

        return p


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
