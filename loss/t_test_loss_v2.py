"""
This file implements another version of t-test loss which is derived from the original version of t-test loss
implemented in t_test_loss.py
This version implements what the authors actually did
"""
import numpy as np
import torch
import torch.nn as nn


class TTestLossV2(nn.Module):
    def __init__(self, beta=0.8, lambda_p=1, lambda_n=0.1):
        super(TTestLossV2, self).__init__()

        self.beta = beta
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n

        return

    def forward(self, residues, image_level_labels, pixel_level_labels, logger=None):
        assert len(residues.shape) == 4  # shape: B, C, H, W
        assert len(image_level_labels.shape) == 2  # shape: B, 1

        if len(pixel_level_labels.shape) == 3:
            pixel_level_labels = pixel_level_labels.unsqueeze(dim=1)
        assert residues.shape == pixel_level_labels.shape

        image_level_labels_np = image_level_labels.cpu().numpy().squeeze()

        positive_idx = np.argwhere(image_level_labels_np == 1)
        num_positive_patch = positive_idx.shape[0]
        positive_idx = torch.from_numpy(positive_idx.squeeze()).long().cuda()

        negative_idx = np.argwhere(image_level_labels_np == 0)
        num_negative_patch = negative_idx.shape[0]
        negative_idx = torch.from_numpy(negative_idx.squeeze()).long().cuda()

        loss = torch.FloatTensor([0]).cuda()

        if num_positive_patch > 0:
            positive_residues = residues.index_select(0, positive_idx)
            positive_pixel_level_labels = pixel_level_labels.index_select(0, positive_idx)
            positive_pixel_level_labels = positive_pixel_level_labels.float()

            positive_residues_masked = positive_residues * positive_pixel_level_labels
            positive_residues_masked_summed_by_image_level = positive_residues_masked.sum(2).sum(2)

            positive_pixel_num_summed_by_image_level = positive_pixel_level_labels.sum(2).sum(2)

            positive_residues_masked_averaged_by_image_level = positive_residues_masked_summed_by_image_level \
                                                               / positive_pixel_num_summed_by_image_level

            mean_residues_positive = positive_residues_masked_averaged_by_image_level.mean()

            loss += torch.max(self.beta - mean_residues_positive, torch.FloatTensor([0]).cuda()) \

            if num_positive_patch > 1:
                var_residues_positive = positive_residues_masked_averaged_by_image_level.var()

                loss += self.lambda_p * var_residues_positive

        if num_negative_patch > 0:
            negative_residues = residues.index_select(0, negative_idx)

            negative_residues_averaged_by_image_level = negative_residues.mean(dim=2).mean(dim=2)

            mean_residues_negative = negative_residues_averaged_by_image_level.mean()

            loss += mean_residues_negative

            if num_negative_patch > 1:
                var_residues_negative = negative_residues_averaged_by_image_level.var()

                loss += self.lambda_n * var_residues_negative

        log_message = 'num_p: {}, num_n: {}, m_r_p: {:.4f}, m_r_n: {:.4f}, v_r_p: {:.4f}, v_r_n: {:.4f}, loss: {:.4f}'.format(
            num_positive_patch,
            num_negative_patch,
            mean_residues_positive.item() if num_positive_patch > 0 else -1,
            mean_residues_negative.item() if num_negative_patch > 0 else -1,
            var_residues_positive.item() if num_positive_patch > 1 else -1,
            var_residues_negative.item() if num_negative_patch > 1 else -1,
            loss.item())

        if logger is not None:
            logger.write_and_print(log_message)
        else:
            print(log_message)

        return loss

    def get_name(self):
        return 'TTestLossV2'
