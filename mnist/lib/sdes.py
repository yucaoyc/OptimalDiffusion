# Copied from https://github.com/CW-Huang/sdeflow-light/blob/main/lib/sdes.py
# with modifications.

import torch
from lib.utils import sample_v, log_normal, sample_vp_truncated_q
import numpy as np

class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max-self.beta_min)*t

    def mean_weight(self, t):
        # the contraction rate for mean
        return torch.exp(-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)

    """
    cov(X_t) = mean_weight(t)**2 * cov(X_0) + var(t) * Id
    """
    def var(self, t):
        # the contraction rate for var
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min)

    def f(self, t, y):
        return - 0.5 * self.beta(t) * y

    def g(self, t, y):
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def sample(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t) * y0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)


class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, drift_a, T, h = None, h_name = "cst-1", vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias
        self.h = h
        self.h_name = h_name
        self.beta_0 = base_sde.beta_min
        self.beta_1 = base_sde.beta_max

    # notes: a = g * score
    # Drift
    def mu(self, t, y, lmbd=0.):
        g_value = self.base_sde.g(self.T-t, y)
        value = (1. - 0.5 * lmbd) * ((self.h(self.T-t, y, self.T)**2 + g_value**2)/2) / g_value
        value *= self.a(y, self.T - t.squeeze())
        value -= self.base_sde.f(self.T - t, y)
        return value

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.h(self.T-t, y, self.T)

    @torch.enable_grad()
    def dsm(self, x, w=lambda z : torch.ones_like(z)):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        y, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(y, t_.squeeze())

        return (w(std) * (a * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2


    def ei(self, y0, t, dt, alpha):
        """
            Exponential integrator for the generative process where h = alpha g.
            alpha is a scalar
            t, dt are scalars.
            This function only propagate the dynamics for one step forward.
        """
        beta_0 = self.beta_0
        beta_1 = self.beta_1
        t_ = (t * torch.ones([y0.size(0), ] + [1 for _ in range(y0.ndim - 1)])).to(y0.device)
        score = self.a(y0, self.T - t_.squeeze())/(self.base_sde.g(self.T-t_, y0) + 1.0e-6)
        gamma = torch.exp(dt/4 * (2 * beta_0 + (2*t - 2 * self.T + dt)*(beta_0 - beta_1)))
        return gamma * y0 + (1+alpha**2)*(gamma - 1) * score + \
            alpha * torch.sqrt(gamma**2 - 1) * torch.randn_like(y0)

    def em(self, y0, t, dt, alpha):
        """
            Euler-Maruyama integrator.
        """
        t_ = (t * torch.ones([y0.size(0), ] + [1 for _ in range(y0.ndim - 1)])).to(y0.device)
        g_value = self.base_sde.g(self.T-t, y0)
        score = self.a(y0, self.T - t_.squeeze())/(g_value + 1.0e-6)
        mu = (1+alpha**2) * (g_value**2/2) * score
        mu -= self.base_sde.f(self.T - t, y0)
        return y0 + dt * mu + alpha * g_value * np.sqrt(dt) * torch.randn_like(y0)
