import numpy as np
import torch


class MetricsImageLevelClassification(object):
    def __init__(self, image_size):
        """
        :param image_size: used for generating image-level mask
        """
        assert isinstance(image_size, list)
        assert len(image_size) == 2

        self.image_size = image_size

        # record metric on validation set for determining the best model to be saved
        self.determine_saving_metric_on_validation_list = list()

        return

    def metric_batch_level(self, preds, labels):
        """
        evaluate at batch-level
        :param preds: probability of each class, shape:[B, 2]
        :param labels: image-level label, shape:[B, 1]
        :return: TPs, TNs, FPs, FNs
        """

        assert len(preds.shape) == 2  # shape: B, C
        assert len(labels.shape) == 1  # shape: B

        # transform preds into binary coding
        if preds.shape[1] > 1:
            _, binary_preds = preds.max(dim=1)
        else:
            binary_preds = torch.zeros_like(preds)
            binary_preds[preds >= 0.5] = 1
            binary_preds = binary_preds.view(-1).long()

        # transfer the tensor into cpu device
        if binary_preds.device.type != 'cpu':
            binary_preds = binary_preds.cpu().detach()
        # transform the tensor into ndarray format
        binary_preds = binary_preds.numpy()

        # transfer the tensor into cpu device
        if labels.device.type != 'cpu':
            labels = labels.cpu()
        # transform the tensor into ndarray format
        labels = labels.numpy()

        assert binary_preds.shape[0] == labels.shape[0]

        image_level_mask_list = list()
        classification_flag_list = list()

        assert binary_preds.shape == labels.shape  # shape: B, H, W

        for patch_idx in range(binary_preds.shape[0]):
            binary_pred = binary_preds[patch_idx]
            label = labels[patch_idx]

            image_level_mask, classification_flag = self.metric_patch_level(binary_pred, label)

            image_level_mask_list.append(image_level_mask)
            classification_flag_list.append(classification_flag)

        image_level_masks_np = np.array(image_level_mask_list)  # shape: B, 1, 1
        classification_flag_np = np.array(classification_flag_list)
        TPs_batch_level = (classification_flag_np == 0).sum()
        TNs_batch_level = (classification_flag_np == 1).sum()
        FPs_batch_level = (classification_flag_np == 2).sum()
        FNs_batch_level = (classification_flag_np == 3).sum()

        return image_level_masks_np, classification_flag_np, TPs_batch_level, TNs_batch_level, FPs_batch_level, \
               FNs_batch_level

    def metric_patch_level(self, pred, label):
        """
        evaluate at patch-level
        :param pred: probability of each class
        :param label: image-level label
        :return: image_level_mask,
                 classification_flag: 0 -> TP, 1 -> TN, 2 -> FP, 3 -> FN
        """
        assert pred == 0 or 1
        assert label == 0 or 1

        classification_flag = 0

        if pred == 1 and label == 1:
            classification_flag = 0
        elif pred == 0 and label == 0:
            classification_flag = 1
        elif pred == 1 and label == 0:
            classification_flag = 2
        elif pred == 0 and label == 1:
            classification_flag = 3

        image_level_mask = np.zeros(self.image_size)
        if pred == 1:
            image_level_mask = np.ones(self.image_size)

        return image_level_mask, classification_flag
