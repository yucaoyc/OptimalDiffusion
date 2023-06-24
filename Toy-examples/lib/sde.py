#! -*-coding:utf-8 -*-

import math
import torch
import numpy as np
import torch.nn as nn

from .net import SimpleNet
from .utils import visualize_line


class SDE(nn.Module):

    def __init__(self, T: float=1.0, beta0: float=1e-4, beta1: float=2e-2, sde_type: str="vp", beta_type: str="linear", data_dim: int=1, hidden_dim: int=32) -> None:
        super().__init__()

        self.t_emb_type = "sinusoidal"

        self.T = T
        self.beta0 = beta0
        self.beta1 = beta1

        self.data_dim = data_dim
        self.time_dim = 2 * math.ceil(data_dim / 2)

        self.sde_type = sde_type
        self.beta_type = beta_type

        self.BT = self.Int_beta(self.T)

        self.score_net = SimpleNet(data_dim = self.data_dim, time_dim = self.time_dim, hidden_dim=hidden_dim)

    def time_embedding(self, t: int | float | np.ndarray):

        if isinstance(t, int):
            t = np.array(t)

        elif isinstance(t, float):
            t = np.array(t)

        elif isinstance(t, np.ndarray):
            pass

        else:
            raise ValueError

        t = t.reshape(-1, 1)

        if self.t_emb_type == "linear":

            t_emb = 2 * t / self.T - 1
            t_emb = t_emb.repeat(self.time_dim, axis=1)

        elif self.t_emb_type == "sinusoidal":

            half = self.time_dim // 2

            w = [np.power(1000, -i / half) for i in range(half)]
            w = np.array(w).reshape(1, -1)

            t_emb = np.zeros(shape=(t.shape[0], 2 * half))

            t_emb[:, ::2] = np.sin(w * t)
            t_emb[:, 1::2] = np.cos(w * t)
            # t_emb[:, :half] = np.sin(w * t)
            # t_emb[:, half:] = np.cos(w * t)

        else:
            raise NotImplementedError
        
        t_emb = torch.from_numpy(t_emb)

        return t_emb

    def beta(self, t: float):

        if self.beta_type == "linear":

            b = self.beta0 + t / self.T * (self.beta1 - self.beta0)

        else:
            raise NotImplementedError

        b = torch.tensor(b)

        return b

    def Int_beta(self, t: float):

        if self.beta_type == "linear":

            Int = self.beta0 * t + (self.beta1 - self.beta0) / self.T / 2 * t * t

        else:
            raise NotImplementedError

        Int = torch.tensor(Int)

        return Int

    def np_Int_beta(self, t: float):

        if self.beta_type == "linear":

            Int = self.beta0 * t + (self.beta1 - self.beta0) / self.T / 2 * t * t

        else:
            raise NotImplementedError

        return Int


    def forward_sde(self, x: torch.Tensor, t: float, eps: torch.FloatTensor=None, to_numpy: bool=False):

        if eps is None:
            eps = torch.randn_like(x)
            
        if self.sde_type == "vp":

            int_beta = self.Int_beta(t)

            m = torch.exp(-int_beta / 2).view(-1, 1).mul(x)
            v = 1 - torch.exp(-int_beta)

            x_t = m + torch.sqrt(v).view(-1, 1).mul(eps)

        else:
            raise NotImplementedError

        if to_numpy:
            x_t = x_t.numpy()

        return x_t

    def get_loss(self, x: torch.Tensor, t: int | np.ndarray):

        t_emb = self.time_embedding(t)

        eps = torch.randn_like(x)
        
        if self.sde_type == "vp":

            int_beta = self.Int_beta(t)

            m = torch.exp(-int_beta / 2).view(-1, 1).mul(x)
            v = 1 - torch.exp(-int_beta)

            x_t = m + torch.sqrt(v).view(-1, 1).mul(eps)

            score_input = torch.cat([x_t, t_emb], dim=1)
            score_ = self.score_net(score_input)

            loss = torch.pow(score_ + eps, 2.0).sum(dim=1, keepdim=True)

        else:
            raise NotImplementedError

        return loss

    def estimate_score(self, data_iter, steps: int=100, lr: float=1e-2, log_every=1000):

        opt = torch.optim.AdamW(self.score_net.parameters(), lr=lr, betas=(0.5, 0.999))

        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size= 2 * steps // 5, gamma=0.5)

        train_loss = None
        train_loss_ema = 0.9

        train_loss_lst = list()

        for train_step in range(steps):

            self.train()

            x = next(data_iter)
            x = torch.from_numpy(x).view(-1, self.data_dim)

            t_0 = 0 * self.T
            t_1 = 1 * self.T
            t = np.random.uniform(low=t_0, high=self.T * t_1, size=(x.size(0),))

            opt.zero_grad()

            loss = self.get_loss(x=x, t=t)
            loss.mean().backward()

            opt.step()

            if train_step == 0:
                train_loss = loss.mean().item()
            else:
                train_loss = train_loss_ema * train_loss + (1 - train_loss_ema) * loss.mean().item()

            if (train_step + 1) % log_every == 0:

                print(f"Step [{train_step + 1:d}/{steps:d}], lr={lr_schedule.get_lr()[0]:.2e}: loss = {train_loss:.2e}")

            train_loss_lst.append(train_loss)

            lr_schedule.step()

        train_loss_lst = np.array(train_loss_lst)

        visualize_line(train_loss_lst, xaxis=None, yscale="log", title="Train loss", savename=None)

    def reverse_sde(self, x_t: torch.Tensor, t: float, dt: float, score: torch.Tensor, sf_alpha: float=1.0):

        z = torch.randn_like(x_t)
        g = torch.sqrt(self.beta(t)).view(-1, 1)
        h = sf_alpha * g

        x_t = x_t - (torch.square(g) / 2 * x_t + (torch.square(g) + torch.square(h)) / 2 * score) * dt + h * z * np.sqrt(-dt)

        return x_t


    def sample(self, x_t: torch.Tensor, T: float=0, N: int=100, to_numpy: bool=False, sf_alpha: float=4.0, exact_score_fn=None, corrupter=None, eps=1e-2):

        self.eval()

        with torch.no_grad():

            dt = (T - self.T) / N

            if dt < 0:

                for t in np.arange(self.T, T, dt):

                    if exact_score_fn is None:

                        t = np.ones(x_t.size(0), dtype=np.int64) * t

                        t_emb = self.time_embedding(t=t)

                        score_input = torch.cat([x_t, t_emb], dim=1)
                        score = self.score_net(score_input) / torch.sqrt(1 - torch.exp(-self.Int_beta(t))).view(-1, 1)

                    else:

                        Bt = self.np_Int_beta(t)
                        score = exact_score_fn(x=x_t.numpy(), Bt=Bt)

                        if corrupter is not None:
                            score = corrupter.err_fun(score, t, eps=eps)

                    x_t = self.reverse_sde(x_t, t, dt=dt, score=score, sf_alpha=sf_alpha)

        if to_numpy:
            x_t = x_t.numpy()

        return x_t
