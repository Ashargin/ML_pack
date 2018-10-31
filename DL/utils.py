import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        mse = nn.MSELoss()
        loss = torch.sqrt(mse(x, y))
        return loss
