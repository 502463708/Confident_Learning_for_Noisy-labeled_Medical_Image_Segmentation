"""
This file implements the original version of t-test loss which is described in the cvpr 2019 paper
"""
import torch
import torch.nn as nn


class TTestLoss(nn.Module):
    def __init__(self, beta=0.8, lambda_p=1, lambda_n=0.1):
        super(TTestLoss, self).__init__()

        self.beta = beta
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n

        return

    def forward(self, residues, image_level_labels, logger=None):
        assert len(residues.shape) == 4  # shape: B, C, H, W
        assert len(image_level_labels.shape) == 2  # shape: B, 1

        image_level_labels = image_level_labels.bool()

        residues_summed_by_image_level = residues.sum(dim=2).sum(dim=2)

        residues_positive = torch.masked_select(residues_summed_by_image_level, image_level_labels)
        residues_negative = torch.masked_select(residues_summed_by_image_level, ~image_level_labels)

        num_positive_patch = residues_positive.shape[0]
        num_negative_patch = residues_negative.shape[0]

        mean_residues_positive = residues_positive.mean()
        mean_residues_negative = residues_negative.mean()

        var_residues_positive = residues_positive.var()
        var_residues_negative = residues_negative.var()

        log_message = 'num_p: {}, num_n: {}, m_r_p: {:.4f}, m_r_n: {:.4f}, v_r_p: {:.4f}, v_r_n: {:.4f}'.format(
            num_positive_patch,
            num_negative_patch,
            mean_residues_positive.item(),
            mean_residues_negative.item(),
            var_residues_positive.item(),
            var_residues_negative.item())

        if logger is not None:
            logger.write_and_print(log_message)
        else:
            print(log_message)

        loss = torch.FloatTensor([0]).cuda()

        if num_positive_patch > 0:
            loss += torch.max(self.beta - mean_residues_positive, torch.FloatTensor([0]).cuda()) \
                    + self.lambda_p * var_residues_positive

        if num_negative_patch > 0:
            loss += mean_residues_negative + self.lambda_n * var_residues_negative

        return loss

    def get_name(self):
        return 'TTestLoss'
