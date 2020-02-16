"""
This file implements a class which can evaluate the recall and false positive
"""
import numpy as np
import torch

from common.utils import get_min_distance, post_process_residue
from skimage import measure


class MetricsReconstruction(object):
    def __init__(self, prob_threshold, area_threshold, distance_threshold, slack_for_recall=False):
        """
        :param prob_threshold: the threshold for the binary process of the residue
        :param area_threshold: the threshold for discarding some connected components
        :param distance_threshold: the threshold for discriminating recall amd FP
        :param slack_for_recall: the bool variable for slacking recall metric standard
        """

        assert 0 <= prob_threshold <= 1
        self.prob_threshold = prob_threshold

        assert area_threshold >= 0
        self.area_threshold = area_threshold

        assert distance_threshold > 0
        self.distance_threshold = distance_threshold

        assert isinstance(slack_for_recall, bool)
        self.slack_for_recall = slack_for_recall

        # record metric on validation set for determining the best model to be saved
        self.determine_saving_metric_on_validation_list = list()

        return

    def metric_batch_level(self, preds, labels):
        """
        evaluate at batch-level
        :param preds: residues
        :param labels: pixel-level label without dilated
        :return: the number of recalled calcification and FP
        """

        assert len(preds.shape) == 4  # shape: B, C, H, W
        assert len(labels.shape) == 3  # shape: B, H, W

        # transfer the tensor into cpu device
        if torch.is_tensor(preds):
            if preds.device.type != 'cpu':
                preds = preds.cpu().detach()
            # transform the tensor into ndarray format
            preds = preds.numpy()

        # transfer the tensor into cpu device
        if torch.is_tensor(labels):
            if labels.device.type != 'cpu':
                labels = labels.cpu()
            # transform the tensor into ndarray format
            labels = labels.numpy()

        # discard the channel dimension
        preds = preds.squeeze(axis=1)

        assert preds.shape == labels.shape  # shape: B, H, W

        post_process_preds_list = list()
        calcification_num_batch_level = 0
        recall_num_batch_level = 0
        FP_num_batch_level = 0
        result_flag_list = list()

        assert preds.shape == labels.shape  # shape: B, H, W

        for patch_idx in range(preds.shape[0]):
            pred = preds[patch_idx, :, :]
            label = labels[patch_idx, :, :]

            # this flag indicates the result of this patch, ranging from 0 - 3
            result_flag = 0

            post_process_pred, calcification_num_patch_level, recall_num_patch_level, FP_num_patch_level = \
                self.metric_patch_level(pred, label)

            post_process_preds_list.append(post_process_pred)
            calcification_num_batch_level += calcification_num_patch_level
            recall_num_batch_level += recall_num_patch_level
            FP_num_batch_level += FP_num_patch_level

            # result_flag = 0 -> TP only;
            #               1 -> FP only;
            #               2 -> FN only;
            #               3 -> both FN & FP
            if FP_num_patch_level > 0:
                result_flag += 1
            if recall_num_patch_level < calcification_num_patch_level:
                result_flag += 2
            result_flag_list.append(result_flag)

        post_process_preds_np = np.array(post_process_preds_list)  # shape: B, H, W

        assert post_process_preds_np.shape == preds.shape

        return post_process_preds_np, calcification_num_batch_level, recall_num_batch_level, FP_num_batch_level, \
               result_flag_list

    def metric_patch_level(self, pred, label):
        assert len(pred.shape) == 2
        assert pred.shape == label.shape

        # post-process residue
        post_process_pred = post_process_residue(pred, self.prob_threshold, self.area_threshold)

        # extract connected components
        post_process_pred_connected_components = measure.label(post_process_pred, connectivity=2)
        label_connected_components = measure.label(label, connectivity=2)

        # analyze properties of each connected component
        post_process_pred_props = measure.regionprops(post_process_pred_connected_components)
        label_props = measure.regionprops(label_connected_components)

        # detected
        pred_list = list()
        pred_num = len(post_process_pred_props)
        if pred_num > 0:
            for idx in range(pred_num):
                pred_list.append(np.array(post_process_pred_props[idx].centroid))

        # annotated
        label_list = list()
        label_num = len(label_props)
        if label_num > 0:
            for idx in range(label_num):
                label_list.append(np.array(label_props[idx].centroid))

        calcification_num = label_num
        recall_num = 0
        FP_num = 0

        # for the negative patch case
        if label_num == 0:
            FP_num = pred_num

        # for the positive patch case with failing to detect anything
        elif pred_num == 0:
            recall_num = 0

        # for the positive patch case with something being detected
        else:
            # calculate recall & FP
            recall_record_np = np.zeros(label_num)
            TP_record_np = np.zeros(pred_num)
            for label_idx in range(label_num):
                for pred_idx in range(pred_num):
                    if np.linalg.norm(label_list[label_idx] - pred_list[pred_idx]) <= self.distance_threshold:
                        recall_record_np[label_idx] = 1
                        TP_record_np[pred_idx] = 1
                    elif self.slack_for_recall:
                        one_connected_component_mask = np.zeros_like(pred)
                        one_connected_component_mask[post_process_pred_connected_components == pred_idx + 1] = 1
                        min_distance = get_min_distance(one_connected_component_mask, label_list[label_idx])
                        if min_distance is not None and min_distance <= self.distance_threshold / 2:
                            recall_record_np[label_idx] = 1
                            TP_record_np[pred_idx] = 1
            recall_num = int(recall_record_np.sum())
            FP_num = int(pred_num - TP_record_np.sum())

        return post_process_pred, calcification_num, recall_num, FP_num
