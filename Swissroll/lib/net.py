#! -*-coding:utf-8 -*-

from torch import nn

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
