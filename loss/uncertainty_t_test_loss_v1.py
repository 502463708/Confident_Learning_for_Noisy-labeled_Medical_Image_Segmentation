"""
This file implements a modified version of t-test loss which is derived from the
implemented in t_test_loss_v3.py
This version implements the weighted t-test loss
"""
import torch
import torch.nn as nn


class UncertaintyTTestLossV1(nn.Module):
    def __init__(self, beta=0.8, lambda_p=1, lambda_n=0.1, u_low=0.02, u_up=0.1, w_low=0.2, w_up=0.8):
        super(UncertaintyTTestLossV1, self).__init__()

        self.beta = beta
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n

        assert 0 <= u_low <= 1
        assert 0 <= u_up <= 1
        assert 0 <= w_low <= 1
        assert 0 <= w_up <= 1

        self.u_low = u_low
        self.u_up = u_up
        self.w_low = w_low
        self.w_up = w_up

        self.k = -(w_up - w_low) / (u_up - u_low)
        self.b = w_low - self.k * u_up

        return

    def forward(self, residues, pixel_level_labels, uncertainty_maps, logger=None):
        assert torch.is_tensor(residues)
        assert torch.is_tensor(pixel_level_labels)
        assert torch.is_tensor(uncertainty_maps)
        assert len(residues.shape) == 4  # shape: B, C, H, W

        # add the missing dimension of channel in pixel_level_labels
        if len(pixel_level_labels.shape) == 3:  # shape: B, H, W
            pixel_level_labels = pixel_level_labels.unsqueeze(dim=1)
        assert residues.shape == pixel_level_labels.shape

        # add the missing dimension of channel in uncertainty_maps
        if len(uncertainty_maps.shape) == 3:  # shape: B, H, W
            uncertainty_maps = uncertainty_maps.unsqueeze(dim=1)
        assert residues.shape == uncertainty_maps.shape

        # pixel_level_labels must be a tensor on gpu devices
        if pixel_level_labels.device.type != 'cuda':
            pixel_level_labels = pixel_level_labels.cuda()

        # uncertainty_maps must be a tensor on gpu devices
        if uncertainty_maps.device.type != 'cuda':
            uncertainty_maps = uncertainty_maps.cuda()

        # reshaped into vectors
        residues = residues.view(-1)
        pixel_level_labels = pixel_level_labels.view(-1)
        uncertainty_maps = uncertainty_maps.view(-1)

        # bool variable for the following torch.masked_select() operation
        positive_pixel_idx = pixel_level_labels.bool()
        negative_pixel_idx = ~positive_pixel_idx

        # split residues into positive and negative one
        positive_residue_pixels = torch.masked_select(residues, positive_pixel_idx)
        negative_residue_pixels = torch.masked_select(residues, negative_pixel_idx)

        # split uncertainty_maps into positive and negative one
        # positive_uncertainty_maps = torch.masked_select(uncertainty_maps, positive_pixel_idx)
        negative_uncertainty_maps = torch.masked_select(uncertainty_maps, negative_pixel_idx)

        loss = torch.FloatTensor([0]).cuda()

        if positive_residue_pixels.shape[0] > 0:
            mean_residue_pixels_positive = positive_residue_pixels.mean()
            loss += torch.max(self.beta - mean_residue_pixels_positive, torch.FloatTensor([0]).cuda())
            # calculate variance only when the number of the positive pixels > 1
            if positive_residue_pixels.shape[0] > 1:
                var_residue_pixels_positive = positive_residue_pixels.var()
                loss += self.lambda_n * var_residue_pixels_positive

        if negative_residue_pixels.shape[0] > 0:
            # generate negative_weights based on negative_uncertainty_maps
            negative_uncertainty_maps_lower_idx = negative_uncertainty_maps < self.u_low
            negative_uncertainty_maps_upper_idx = negative_uncertainty_maps > self.u_up
            negative_weights = self.k * negative_uncertainty_maps + self.b
            negative_weights[negative_uncertainty_maps_lower_idx] = 1
            negative_weights[negative_uncertainty_maps_upper_idx] = self.w_low

            mean_residue_pixels_negative = (negative_residue_pixels * negative_weights).mean()
            loss += mean_residue_pixels_negative
            # calculate variance only when the number of the negative pixels > 1
            if negative_residue_pixels.shape[0] > 1:
                var_residue_pixels_negative = negative_residue_pixels.var()
                loss += self.lambda_p * var_residue_pixels_negative

        log_message = 'num_p: {}, num_n: {}, m_r_p: {:.4f}, m_r_n: {:.4f}, v_r_p: {:.4f}, v_r_n: {:.4f}, loss: {:.4f}'.format(
            positive_residue_pixels.shape[0],
            negative_residue_pixels.shape[0],
            mean_residue_pixels_positive.item() if positive_residue_pixels.shape[0] > 0 else -1,
            mean_residue_pixels_negative.item() if negative_residue_pixels.shape[0] > 0 else -1,
            var_residue_pixels_positive.item() if positive_residue_pixels.shape[0] > 1 else -1,
            var_residue_pixels_negative.item() if negative_residue_pixels.shape[0] > 1 else -1,
            loss.item())

        if logger is not None:
            logger.write_and_print(log_message)
        else:
            print(log_message)

        return loss

    def get_name(self):
        return 'UncertaintyTTestLossV1'
