"""
This file implements
"""
import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

        self.loss_func = torch.nn.L1Loss()

        return

    def forward(self, preds, labels, logger=None):
        assert torch.is_tensor(preds)
        assert torch.is_tensor(labels)
        assert labels.shape[0] == preds.shape[0]

        # labels must be a tensor on gpu devices
        if labels.device.type != 'cuda':
            labels = labels.cuda()

        labels = labels.float()

        # reshape preds and labels into vectors
        preds = preds.view(-1)
        labels = labels.view(-1)

        loss = self.loss_func(preds, labels)

        return loss

    def get_name(self):

        return 'L1Loss'
