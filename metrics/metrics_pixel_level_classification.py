"""
This file implements a class which can evaluate the recall and false positive
"""
import numpy as np
import torch


class MetricsPixelLevelClassification(object):
    def __init__(self, num_classes):
        # record metric on validation set for determining the best model to be saved
        self.determine_saving_metric_on_validation_list = list()

        assert num_classes >= 2
        self.num_classes = num_classes

        return

    def metric_batch_level(self, preds, labels):
        """
        evaluate at batch-level
        :param preds: classification results after softmax
        :param labels: pixel-level label without dilated
        :return: the number of recalled calcification and FP
        """

        assert len(preds.shape) == 4  # shape: B, C, H, W
        assert len(labels.shape) == 3  # shape: B, H, W

        _, post_process_preds = torch.max(preds, dim=1)

        # transfer the tensor into cpu device
        if torch.is_tensor(post_process_preds):
            if post_process_preds.device.type != 'cpu':
                post_process_preds = post_process_preds.cpu().detach()
            # transform the tensor into ndarray format
            post_process_preds = post_process_preds.numpy()

        # transfer the tensor into cpu device
        if torch.is_tensor(labels):
            if labels.device.type != 'cpu':
                labels = labels.cpu()
            # transform the tensor into ndarray format
            labels = labels.numpy()

        assert post_process_preds.shape == labels.shape  # shape: B, H, W

        dice_batch_level = list()

        for patch_idx in range(post_process_preds.shape[0]):
            post_process_pred = post_process_preds[patch_idx, :, :]
            label = labels[patch_idx, :, :]

            dice_patch_level = self.metric_patch_level(post_process_pred, label)

            dice_batch_level.append(dice_patch_level)

        dice_batch_level = np.array(dice_batch_level).mean(axis=0)

        return post_process_preds, dice_batch_level

    def metric_patch_level(self, pred, label):
        assert len(pred.shape) == 2
        assert pred.shape == label.shape

        dice_patch_level = list()

        for class_idx in range(self.num_classes):
            pred_class = pred == class_idx
            label_class = label == class_idx

            pred_class = pred_class.reshape(-1)
            label_class = label_class.reshape(-1)

            intersection = (pred_class * label_class).sum()
            union = pred_class.sum() + label_class.sum()

            score = 2. * intersection / (union + 1e-4)

            dice_patch_level.append(score)

        dice_patch_level = np.array(dice_patch_level)

        return dice_patch_level
