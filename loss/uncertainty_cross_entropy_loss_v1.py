"""
This file implements a modified version of Cross-Entropy loss.
The uncertainty maps are equipped with the ability of transforming
negative samples -> positive samples.
"""
import torch
import torch.nn as nn


class UncertaintyCrossEntropyLossV1(nn.Module):

    def __init__(self, upn, epsilon=0.25, size_average=True):
        super(UncertaintyCrossEntropyLossV1, self).__init__()

        self.upn = upn
        self.epsilon = epsilon
        self.size_average = size_average

        return

    def forward(self, preds, labels, uncertainty_maps, logger=None):
        assert len(preds.shape) == 2
        assert len(uncertainty_maps.shape) == 3

        # reshape labels into vectors
        labels = labels.float()
        labels = labels.view(-1)

        num_samples = labels.shape[0]

        # activation the preds
        preds = torch.softmax(preds, dim=1)

        # calculate image-level uncertainty based on pixel-level uncertainty
        image_level_uncertainties = uncertainty_maps > 0.05
        image_level_uncertainties = image_level_uncertainties.sum(1).sum(1)
        uncertainty_flags = (image_level_uncertainties > self.upn).cuda()
        flags = uncertainty_flags & (labels == 0)

        # smoothing labels by assign epsilon to negative class
        smoothed_labels = labels + self.epsilon
        smoothed_labels[~flags] = labels[~flags]

        class_0_loss = -(1 - smoothed_labels) * torch.log(preds[:, 0])
        class_1_loss = -smoothed_labels * torch.log(preds[:, 1])

        batch_loss = class_0_loss.sum() + class_1_loss.sum()

        if self.size_average:
            loss = batch_loss / num_samples
        else:
            loss = batch_loss

        return loss

    def get_name(self):

        return 'UncertaintyCrossEntropyLossV1'
