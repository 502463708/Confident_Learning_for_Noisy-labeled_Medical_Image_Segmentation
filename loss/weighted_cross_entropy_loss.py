"""
This file implements a modified version of Cross-Entropy loss.
The confident maps soften the labels.
"""
import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        return

    def forward(self, preds, labels, weights, logger=None):
        assert len(preds.shape) == 4
        assert len(labels.shape) == 3
        assert len(weights.shape) == 2

        # activate the preds
        preds = torch.softmax(preds, dim=1)
        preds = preds.permute(0, 2, 3, 1).contiguous()

        num_samples = preds.shape[0]
        assert num_samples > 0

        # reshape labels into vectors
        labels = labels.cuda().long()

        # reshape confident_maps into vectors
        weights = weights.view(-1)

        batch_loss = torch.FloatTensor([0]).cuda()

        for idx in range(num_samples):
            pred = preds[idx].view(-1, 2)
            label = labels[idx].view(-1)
            weight = weights[idx]
            batch_loss += weight * self.cross_entropy_loss(pred, label)

        if self.size_average:
            loss = batch_loss / num_samples
        else:
            loss = batch_loss

        return loss

    def get_name(self):

        return 'WeightedCrossEntropyLoss'
