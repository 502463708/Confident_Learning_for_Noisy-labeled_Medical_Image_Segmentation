import cv2
import numpy as np
import os
import random
import shutil

from skimage import measure


class PatchLevelDatasetSelection:
    def __init__(self, data_root_dir, connected_component_threshold, output_dir, enable_random=False,
                 training_ratio=3, validation_ratio=1, test_ratio=1):
        assert os.path.isdir(data_root_dir), '{} does not exist'.format(data_root_dir)
        assert connected_component_threshold > 0, 'connected_component_threshold shall be a positive number'
        assert isinstance(enable_random, bool), 'enable_random shall be a bool variable'
        assert isinstance(training_ratio, int) and training_ratio > 0, 'training_ratio shall be a positive integer'
        assert isinstance(validation_ratio,
                          int) and validation_ratio > 0, 'validation_ratio shall be a positive integer'
        assert isinstance(test_ratio, int) and test_ratio > 0, 'test_ratio shall be a positive integer'

        self.data_root_dir = data_root_dir
        self.connected_component_threshold = connected_component_threshold
        self.output_dir = output_dir
        self.enable_random = enable_random
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        self.selected_positive_patch_path_list = list()
        self.selected_negative_patch_path_list = list()
        self.selected_positive_patch_num = 0

        self.training_positive_patch_path_list = list()
        self.validation_positive_patch_path_list = list()
        self.test_positive_patch_path_list = list()
        self.training_negative_patch_path_list = list()
        self.validation_negative_patch_path_list = list()
        self.test_negative_patch_path_list = list()

        # create output tree structure dirs
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        #
        patch_class_list = ['positive_patches', 'negative_patches']
        data_class_list = ['training', 'validation', 'test']
        image_class_list = ['images', 'labels']
        #
        os.mkdir(output_dir)
        for patch_class in patch_class_list:
            curr_path = os.path.join(output_dir, patch_class)
            os.mkdir(curr_path)
            for data_class in data_class_list:
                curr_path = os.path.join(output_dir, patch_class, data_class)
                os.mkdir(curr_path)
                for image_class in image_class_list:
                    curr_path = os.path.join(output_dir, patch_class, data_class, image_class)
                    os.mkdir(curr_path)

        return

    def process_positive_patch_dir(self, positive_patch_dir):
        assert os.path.isdir(positive_patch_dir), '{} does not exist'.format(positive_patch_dir)

        positive_patch_filename_list = os.listdir(positive_patch_dir)

        print('currently processing dir: {}, containing {} images'.format(positive_patch_dir,
                                                                          len(positive_patch_filename_list)))

        for positive_patch_filename in positive_patch_filename_list:
            positive_patch_path = os.path.join(positive_patch_dir, positive_patch_filename)
            positive_mask_patch_path = positive_patch_path.replace('images', 'labels')

            # check mask path
            assert os.path.exists(positive_mask_patch_path)
            # load mask
            mask_np = cv2.imread(positive_mask_patch_path, cv2.IMREAD_GRAYSCALE)
            mask_np = mask_np.astype(np.float)
            if mask_np.max() == 255:
                mask_np /= 255.0
            # calculate the number of connected components
            connected_components = measure.label(mask_np, connectivity=2)
            num_connected_components = connected_components.max()

            # add to the list only when the num_connected_components is legal
            if num_connected_components <= self.connected_component_threshold:
                self.selected_positive_patch_path_list.append(positive_patch_path)

        return

    def process_negative_patch_dir(self, negative_patch_dir):
        assert os.path.isdir(negative_patch_dir), '{} does not exist'.format(negative_patch_dir)

        negative_patch_filename_list = os.listdir(negative_patch_dir)

        print('currently processing dir: {}, containing {} images'.format(negative_patch_dir,
                                                                          len(negative_patch_filename_list)))

        for negative_patch_filename in negative_patch_filename_list:
            negative_patch_path = os.path.join(negative_patch_dir, negative_patch_filename)
            self.selected_negative_patch_path_list.append(negative_patch_path)

        return

    def select_patch_path_list(self):
        if not self.enable_random:
            random.seed(0)

        self.selected_positive_patch_num = len(self.selected_positive_patch_path_list)

        self.selected_positive_patch_path_list = random.sample(self.selected_positive_patch_path_list,
                                                               self.selected_positive_patch_num)

        self.selected_negative_patch_path_list = random.sample(self.selected_negative_patch_path_list,
                                                               self.selected_positive_patch_num)

        # split positive (negative) patches into training, validation and test
        idx_1 = int(self.training_ratio * self.selected_positive_patch_num / (
                self.training_ratio + self.validation_ratio + self.test_ratio))

        idx_2 = int((self.training_ratio + self.validation_ratio) * self.selected_positive_patch_num / (
                self.training_ratio + self.validation_ratio + self.test_ratio))

        self.training_positive_patch_path_list = self.selected_positive_patch_path_list[:idx_1]
        self.validation_positive_patch_path_list = self.selected_positive_patch_path_list[idx_1: idx_2]
        self.test_positive_patch_path_list = self.selected_positive_patch_path_list[idx_2:]

        self.training_negative_patch_path_list = self.selected_negative_patch_path_list[:idx_1]
        self.validation_negative_patch_path_list = self.selected_negative_patch_path_list[idx_1: idx_2]
        self.test_negative_patch_path_list = self.selected_negative_patch_path_list[idx_2:]

        print('training set contains {0} positive patches and {0} negative patches'.format(
            len(self.training_positive_patch_path_list)))
        print('validation set contains {0} positive patches and {0} negative patches'.format(
            len(self.validation_positive_patch_path_list)))
        print('test set contains {0} positive patches and {0} negative patches'.format(
            len(self.test_positive_patch_path_list)))

        return

    def copy_images_and_labels(self, image_path_list, dst_dir):
        assert isinstance(image_path_list, list)
        assert os.path.isdir(dst_dir)

        print('copy files into {}'.format(dst_dir))

        for src_image_path in image_path_list:
            filename = src_image_path.split('/')[-1]
            src_mask_path = src_image_path.replace('images', 'labels')

            dst_image_path = os.path.join(dst_dir, 'images', filename)
            dst_mask_path = os.path.join(dst_dir, 'labels', filename)

            shutil.copyfile(src_image_path, dst_image_path)
            shutil.copyfile(src_mask_path, dst_mask_path)

        return

    def run(self):
        # collect all of the positive and negative image paths
        patch_class_list = ['positive_patches', 'negative_patches']
        data_class_list = ['training', 'validation', 'test']
        for patch_class in patch_class_list:
            for data_class in data_class_list:
                curr_path = os.path.join(self.data_root_dir, patch_class, data_class, 'images')
                if patch_class == 'positive_patches':
                    self.process_positive_patch_dir(curr_path)
                else:
                    self.process_negative_patch_dir(curr_path)

        self.select_patch_path_list()

        self.copy_images_and_labels(self.training_positive_patch_path_list,
                                    os.path.join(self.output_dir, 'positive_patches', 'training'))
        self.copy_images_and_labels(self.validation_positive_patch_path_list,
                                    os.path.join(self.output_dir, 'positive_patches', 'validation'))
        self.copy_images_and_labels(self.test_positive_patch_path_list,
                                    os.path.join(self.output_dir, 'positive_patches', 'test'))
        self.copy_images_and_labels(self.training_negative_patch_path_list,
                                    os.path.join(self.output_dir, 'negative_patches', 'training'))
        self.copy_images_and_labels(self.validation_negative_patch_path_list,
                                    os.path.join(self.output_dir, 'negative_patches', 'validation'))
        self.copy_images_and_labels(self.test_negative_patch_path_list,
                                    os.path.join(self.output_dir, 'negative_patches', 'test'))

        return
