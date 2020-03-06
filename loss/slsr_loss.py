"""
This file implements a modified version of Cross-Entropy loss.
The confident maps soften the labels.
"""
import torch
import torch.nn as nn


class SLSRLoss(nn.Module):

    def __init__(self, epsilon=0.25, size_average=True):
        super(SLSRLoss, self).__init__()

        self.epsilon = epsilon
        self.size_average = size_average

        return

    def forward(self, preds, labels, confident_maps, logger=None):
        assert len(preds.shape) == 4
        assert len(labels.shape) == len(confident_maps.shape) == 3
        assert labels.shape == confident_maps.shape

        # activate the preds
        preds = torch.softmax(preds, dim=1)
        preds = torch.clamp(preds, min=1e-4)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, 2)

        num_samples = preds.shape[0]
        assert num_samples > 0

        # reshape labels into vectors
        labels = labels.float()
        labels = labels.view(-1).cuda()

        # reshape confident_maps into vectors
        confident_maps = confident_maps.float()
        confident_maps = confident_maps.view(-1).cuda()

        # smoothing labels
        smoothed_labels = labels
        smoothed_labels[(labels == 0) & (confident_maps == 1)] += self.epsilon
        smoothed_labels[(labels == 1) & (confident_maps == 1)] -= self.epsilon

        class_0_loss = -(1 - smoothed_labels) * torch.log(preds[:, 0])
        class_1_loss = -smoothed_labels * torch.log(preds[:, 1])

        batch_loss = class_0_loss.sum() + class_1_loss.sum()

        if self.size_average:
            loss = batch_loss / num_samples
        else:
            loss = batch_loss

        return loss

    def get_name(self):

        return 'SLSRLoss'
