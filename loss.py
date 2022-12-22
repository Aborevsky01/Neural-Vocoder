import torch
import torch.nn as nn


class GanLoss(nn.Module):
    def __init__(self, phase):
        super().__init__()
        if phase == 'G':
            self.loss_func = lambda x: sum([torch.mean((1 - fake) ** 2) for fake in x])
        elif phase == 'D':
            self.loss_func = lambda x: sum([torch.mean((real - 1) ** 2) for real in x[0]]) + \
                                       sum([torch.mean(fake ** 2) for fake in x[1]])

    def forward(self, x):
        return self.loss_func(x)


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, y, x):
        return sum(
            [torch.mean(self.criterion(fake, real)) for sub_x, sub_y in zip(x, y) for fake, real in zip(sub_x, sub_y)])
