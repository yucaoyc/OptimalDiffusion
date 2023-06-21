#! -*-coding:utf-8 -*-

import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

class Block1D(nn.Module):

    def __init__(self, in_dim, out_dim, out_act=False) -> None:
        super().__init__()

        self.linear = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

        self.out_act = nn.ReLU() if out_act else nn.Identity()

    def forward(self, x):

        y = self.act(self.linear(x))
        y = y + x

        y = self.out_act(self.out(y))

        return y

class SimpleNet(nn.Module):

    def __init__(self, data_dim: int=1, time_dim: int=0, hidden_dim: int=32) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim, bias=True), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim, bias=True), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, data_dim, bias=True), 
        )

    def forward(self, x):

        return self.model(x)
