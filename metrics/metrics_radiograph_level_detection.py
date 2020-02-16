"""
This file implements a class which can evaluate the recall and false positive
"""
import copy
import numpy as np

from skimage import measure


class DetectionResultRecord(object):
    def __init__(self, score_threshold_stride):
        assert score_threshold_stride > 0
        self.score_threshold_stride = score_threshold_stride

        self.calcification_num_dataset_level = 0

        detected_num_dataset_level_list = list()
        recall_num_dataset_level_list = list()
        FP_num_dataset_level_list = list()

        score_threshold = 0
        while score_threshold <= 1:
            detected_num_dataset_level_list.append(0)
            recall_num_dataset_level_list.append(0)
            FP_num_dataset_level_list.append(0)

            score_threshold += self.score_threshold_stride

        self.recall_num_dataset_level_np = np.array(recall_num_dataset_level_list)
        self.FP_num_dataset_level_np = np.array(FP_num_dataset_level_list)
        self.detected_num_dataset_level_np = np.array(detected_num_dataset_level_list)

        return

    def __add__(self, other):
        assert self.score_threshold_stride == other.score_threshold_stride

        self.calcification_num_dataset_level += other.calcification_num_dataset_level
        self.detected_num_dataset_level_np += other.detected_num_dataset_level_np
        self.recall_num_dataset_level_np += other.recall_num_dataset_level_np
        self.FP_num_dataset_level_np += other.FP_num_dataset_level_np

        return self

    def update_calcification_num(self, calcification_num_dataset_level):
        self.calcification_num_dataset_level += calcification_num_dataset_level

        return

    def update_detected_num(self, threshold_idx, detected_num):
        self.detected_num_dataset_level_np[threshold_idx] += detected_num

        return

    def update_recall_num(self, threshold_idx, recall_num):
        self.recall_num_dataset_level_np[threshold_idx] += recall_num

        return

    def update_FP_num(self, threshold_idx, FP_num):
        self.FP_num_dataset_level_np[threshold_idx] += FP_num

        return

    def print(self, logger=None):
        if logger is None:
            print('The annotated calcification number is: {}'.format(self.calcification_num_dataset_level))
            score_threshold = 0
            threshold_idx = 0
            while score_threshold <= 1:
                print('score_threshold = {:.4f}, detected_number = {}, recall_number = {}, FP_number = {}'.
                      format(score_threshold,
                             self.detected_num_dataset_level_np[threshold_idx],
                             self.recall_num_dataset_level_np[threshold_idx],
                             self.FP_num_dataset_level_np[threshold_idx]))

                score_threshold += self.score_threshold_stride
                threshold_idx += 1
        else:
            logger.write_and_print(
                'The annotated calcification number is: {}'.format(self.calcification_num_dataset_level))
            score_threshold = 0
            threshold_idx = 0
            while score_threshold <= 1:
                logger.write_and_print(
                    'score_threshold = {:.4f}, detected_number = {}, recall_number = {}, FP_number = {}'.
                    format(score_threshold,
                           self.detected_num_dataset_level_np[threshold_idx],
                           self.recall_num_dataset_level_np[threshold_idx],
                           self.FP_num_dataset_level_np[threshold_idx]))

                score_threshold += self.score_threshold_stride
                threshold_idx += 1

        return


class MetricsRadiographLevelDetection(object):
    def __init__(self, distance_threshold, score_threshold_stride):
        """
        :param distance_threshold: the threshold for discriminating recall amd FP
        """

        assert distance_threshold > 0
        self.distance_threshold = distance_threshold

        assert score_threshold_stride > 0
        self.score_threshold_stride = score_threshold_stride

        self.detection_result_record_dataset_level = DetectionResultRecord(self.score_threshold_stride)

        return

    def metric_all_score_thresholds(self, pred_coord_list, pred_score_list, label_coord_list,
                                    processed_residue_radiograph_np=None):
        """
        evaluate at batch-level
        :param preds: residues
        :param labels: pixel-level label without dilated
        :return: the number of recalled calcification and FP
        """

        assert len(pred_score_list) == len(pred_coord_list)

        detection_result_record_radiograph_level = DetectionResultRecord(self.score_threshold_stride)
        detection_result_record_radiograph_level.update_calcification_num(len(label_coord_list))

        score_threshold = 0
        threshold_idx = 0
        filtered_residue_radiograph_np = copy.copy(processed_residue_radiograph_np)
        while score_threshold <= 1:
            processed_pred_coord_list = self.process_pred_coord_list(pred_coord_list, pred_score_list, score_threshold)

            if processed_residue_radiograph_np is None:
                detection_result_record_radiograph_level = self.metric_a_specific_score_threshold(
                    processed_pred_coord_list,
                    label_coord_list,
                    threshold_idx,
                    detection_result_record_radiograph_level)
            else:
                filtered_residue_radiograph_np = self.filter_residue(filtered_residue_radiograph_np,
                                                                     processed_pred_coord_list)
                detection_result_record_radiograph_level = self.metric_a_specific_score_threshold(
                    processed_pred_coord_list,
                    label_coord_list,
                    threshold_idx,
                    detection_result_record_radiograph_level,
                    filtered_residue_radiograph_np)

            score_threshold += self.score_threshold_stride
            threshold_idx += 1

        self.detection_result_record_dataset_level = self.detection_result_record_dataset_level + \
                                                     detection_result_record_radiograph_level

        return detection_result_record_radiograph_level

    def process_pred_coord_list(self, pred_coord_list, pred_score_list, score_threshold):
        assert len(pred_score_list) == len(pred_coord_list)

        processed_pred_coord_list = list()

        for pred_idx in range(len(pred_coord_list)):
            pred_coord = pred_coord_list[pred_idx]
            pred_score = pred_score_list[pred_idx]
            if pred_score >= score_threshold:
                processed_pred_coord_list.append(pred_coord)

        assert len(pred_score_list) >= len(processed_pred_coord_list)

        return processed_pred_coord_list

    def filter_residue(self, processed_residue_radiograph_np, processed_pred_coord_list):
        assert len(processed_residue_radiograph_np.shape) == 2

        filtered_residue_radiograph_np = copy.copy(processed_residue_radiograph_np)

        residue_mask = np.zeros_like(processed_residue_radiograph_np)
        residue_mask[processed_residue_radiograph_np > 0] = 1
        connected_components = measure.label(residue_mask)
        props = measure.regionprops(connected_components)

        connected_idx = 0
        if len(props) > 0:
            for prop in props:
                connected_idx += 1

                # generate logic indexes for this connected component
                indexes = connected_components == connected_idx

                remove_this_connected_component = True
                for processed_pred_coord in processed_pred_coord_list:
                    if indexes[int(processed_pred_coord[0]), int(processed_pred_coord[1])]:
                        remove_this_connected_component = False

                if remove_this_connected_component:
                    filtered_residue_radiograph_np[indexes] = 0

        assert processed_residue_radiograph_np.shape == filtered_residue_radiograph_np.shape

        return filtered_residue_radiograph_np

    def metric_a_specific_score_threshold(self, detected_coord_list, label_coord_list, threshold_idx,
                                          detection_result_record_radiograph_level,
                                          processed_residue_radiograph_np=None):
        slack_for_recall = False
        height = 0
        width = 0
        if processed_residue_radiograph_np is not None:
            assert len(processed_residue_radiograph_np.shape) == 2

            height = processed_residue_radiograph_np.shape[0]
            width = processed_residue_radiograph_np.shape[1]
            slack_for_recall = True

        detected_num = len(detected_coord_list)
        label_num = len(label_coord_list)

        recall_num = 0
        FP_num = 0

        # for the negative patch case
        if label_num == 0:
            FP_num = detected_num

        # for the positive patch case with failing to detect anything
        elif detected_num == 0:
            recall_num = 0

        # for the positive patch case with something being detected
        else:
            # calculate recall
            for label_idx in range(label_num):
                for detected_idx in range(detected_num):
                    label_coord = label_coord_list[label_idx]
                    detected_coord = detected_coord_list[detected_idx]
                    if np.linalg.norm(label_coord - detected_coord) <= self.distance_threshold:
                        recall_num += 1
                        break
                    elif slack_for_recall:
                        residue_accumulated_value = 0
                        # whether there exists 1 pixel at least with residue value > 0 in the round area which is
                        # centered by label_coord and set radius as self.distance_threshold / 2
                        crop_center_row_idx = label_coord[0]
                        crop_center_column_idx = label_coord[1]
                        crop_row_start_idx = int(np.clip(crop_center_row_idx - self.distance_threshold / 2, 0, height))
                        crop_row_end_idx = int(np.clip(crop_center_row_idx + self.distance_threshold / 2, 0, height))
                        crop_column_start_idx = int(
                            np.clip(crop_center_column_idx - self.distance_threshold / 2, 0, width))
                        crop_column_end_idx = int(
                            np.clip(crop_center_column_idx + self.distance_threshold / 2, 0, width))

                        for row_idx in range(crop_row_start_idx, crop_row_end_idx):
                            for column_idx in range(crop_column_start_idx, crop_column_end_idx):
                                if np.linalg.norm(
                                        np.array([row_idx, column_idx]) - label_coord) <= self.distance_threshold / 2:
                                    residue_accumulated_value += processed_residue_radiograph_np[row_idx, column_idx]

                        if residue_accumulated_value > 0:
                            recall_num += 1
                            break

            # calculate FP
            for detected_idx in range(detected_num):
                for label_idx in range(label_num):
                    if np.linalg.norm(
                            label_coord_list[label_idx] - detected_coord_list[detected_idx]) <= self.distance_threshold:
                        break
                    if label_idx == label_num - 1:
                        FP_num += 1

        detection_result_record_radiograph_level.update_detected_num(threshold_idx, detected_num)
        detection_result_record_radiograph_level.update_recall_num(threshold_idx, recall_num)
        detection_result_record_radiograph_level.update_FP_num(threshold_idx, FP_num)

        return detection_result_record_radiograph_level
