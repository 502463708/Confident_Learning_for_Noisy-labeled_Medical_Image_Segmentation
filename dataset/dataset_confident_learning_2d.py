import cv2
import numpy as np
import os
import random
import torch

from torch.utils.data import Dataset


def generate_and_check_filename_list(dataset_type_dir, class_name):
    # check the correctness of the given path
    assert os.path.isdir(dataset_type_dir)

    print('-------------------------------------------------------------------------------------------------------')
    print('Starting checking the files in {0}...'.format(dataset_type_dir))

    image_dir = os.path.join(dataset_type_dir, 'images')
    label_dir = os.path.join(dataset_type_dir, class_name)

    assert os.path.isdir(image_dir)
    assert os.path.isdir(label_dir)

    filename_list = os.listdir(image_dir)

    for filename in filename_list:
        # each file's extension must be 'png'
        assert filename.split('.')[-1] == 'png'

        assert os.path.exists(os.path.join(image_dir, filename))
        assert os.path.exists(os.path.join(label_dir, filename))

    print('Checking passed: all of the involved {} files are legal with extension.'.format(len(filename_list)))
    print('-------------------------------------------------------------------------------------------------------')

    return filename_list


class ConfidentLearningDataset2d(Dataset):
    def __init__(self, data_root_dir, mode, class_name, enable_random_sampling, image_channels, cropping_size,
                 enable_data_augmentation, enable_vertical_flip=False, enable_horizontal_flip=False):

        super(ConfidentLearningDataset2d, self).__init__()

        # the data root path must exist
        assert os.path.isdir(data_root_dir)

        # the mode must be one of 'training', 'validation' or 'test'
        assert mode in ['training', 'validation', 'test']
        self.mode = mode

        # the class_name must be one of 'clavicle', 'heart' or 'lung'
        assert class_name in ['clavicle', 'heart', 'lung']
        self.class_name = class_name

        # enable_random_sampling must be a bool variable
        assert isinstance(enable_random_sampling, bool)
        self.enable_random_sampling = enable_random_sampling

        # image_channels must be a positive number
        assert image_channels > 0
        self.image_channels = image_channels

        # cropping_size contains length of height and width
        assert len(cropping_size) == 2
        self.cropping_size = cropping_size

        # enable_data_augmentation is a bool variable
        assert isinstance(enable_data_augmentation, bool)
        self.enable_data_augmentation = enable_data_augmentation

        # enable_vertical_flip is a bool variable
        assert isinstance(enable_vertical_flip, bool)
        self.enable_vertical_flip = enable_vertical_flip

        # enable_horizontal_flip is a bool variable
        assert isinstance(enable_horizontal_flip, bool)
        self.enable_horizontal_flip = enable_horizontal_flip

        # the image directory
        self.dataset_type_dir = os.path.join(data_root_dir, self.mode)

        # the image filename list
        self.filename_list = generate_and_check_filename_list(self.dataset_type_dir, self.class_name)

        return

    def __getitem__(self, index):
        """
        :param index
        :return: Tensors
        """
        if self.enable_random_sampling:
            # sample randomly
            filename = random.sample(self.filename_list, 1)[0]
        else:
            # sample an image file
            filename = self.filename_list[index]

        image_path = os.path.join(self.dataset_type_dir, 'images', filename)

        # get the corresponding pixel-level label path
        pixel_level_label_path = image_path.replace('images', self.class_name)

        # check the existence of the sampled patch
        assert os.path.exists(image_path)
        assert os.path.exists(pixel_level_label_path)

        # load image
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_np = image_np.astype(np.float)
        #
        original_shape = image_np.shape
        #
        # normalize the intensity range of image: [0, 255] -> [0, 1]
        image_np /= 255.0

        # load pixel-level label
        pixel_level_label_np = cv2.imread(pixel_level_label_path, cv2.IMREAD_GRAYSCALE)
        pixel_level_label_np = pixel_level_label_np.astype(np.float)
        pixel_level_label_np /= 255.0

        # check the consistency of size between image, its pixel-level label
        assert image_np.shape == pixel_level_label_np.shape

        # implement data augmentation only when the variable enable_data_augmentation is set True
        if self.enable_data_augmentation:
            if self.enable_vertical_flip and random.random() >= 0.5:
                image_np = np.flipud(image_np)
                pixel_level_label_np = np.flipud(pixel_level_label_np)
            if self.enable_horizontal_flip and random.random() >= 0.5:
                image_np = np.fliplr(image_np)
                pixel_level_label_np = np.fliplr(pixel_level_label_np)

        # guarantee image_np, pixel_level_label_np and uncertainty_map keep contiguous after data augmentation
        image_np = np.ascontiguousarray(image_np)
        pixel_level_label_np = np.ascontiguousarray(pixel_level_label_np)

        # convert ndarray to tensor
        #
        # image tensor
        image_tensor = torch.FloatTensor(image_np).unsqueeze(dim=0)  # shape: [C, H, W]
        #
        # pixel-level label tensor
        pixel_level_label_tensor = torch.LongTensor(pixel_level_label_np)  # shape: [H, W]
        #

        # resize images, masks and if the actual size is not consistent with the target size
        if np.linalg.norm(np.array(self.cropping_size) - original_shape) > 1e-3:
            image_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(dim=0),
                                                           size=(self.cropping_size[0], self.cropping_size[1]),
                                                           scale_factor=None, mode='bilinear',
                                                           align_corners=False).squeeze(dim=0)

            pixel_level_label_tensor = pixel_level_label_tensor.float()
            pixel_level_label_tensor = torch.nn.functional.interpolate(
                pixel_level_label_tensor.unsqueeze(dim=0).unsqueeze(dim=0),
                size=(self.cropping_size[0], self.cropping_size[1]),
                scale_factor=None, mode='nearest').squeeze().long()

        assert len(image_tensor.shape) == 3
        assert len(pixel_level_label_tensor.shape) == 2
        assert image_tensor.shape[0] == self.image_channels
        assert image_tensor.shape[1] == pixel_level_label_tensor.shape[0] == self.cropping_size[0]
        assert image_tensor.shape[2] == pixel_level_label_tensor.shape[1] == self.cropping_size[1]

        return image_tensor, pixel_level_label_tensor, filename

    def __len__(self):

        return len(self.filename_list)
