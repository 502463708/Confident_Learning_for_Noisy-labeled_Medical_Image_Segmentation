"""
This file implements
"""
import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

        self.loss_func = nn.L1Loss()

        return

    def forward(self, preds_num, labels_num, logger=None):
        assert torch.is_tensor(preds_num)
        assert torch.is_tensor(labels_num)
        assert len(preds_num.shape) == 2  # shape: B*1
        assert len(labels_num.shape) == 2  # shape: B*1

        # label number must be a tensor on gpu devices
        if labels_num.device.type != 'cuda':
            pixel_level_labels = labels_num.cuda()

        loss = self.loss_func(preds_num, labels_num)

        return loss

    def get_name(self):
        return 'L1Loss'
