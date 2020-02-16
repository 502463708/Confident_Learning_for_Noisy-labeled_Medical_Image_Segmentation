import cv2
import numpy as np
import os
import random
import SimpleITK as sitk
import torch

from common.utils import dilate_image_level_label
from skimage import measure
from torch.utils.data import Dataset


def generate_and_check_filename_list(dataset_type_dir, load_uncertainty_map=False):
    # check the correctness of the given path
    assert os.path.isdir(dataset_type_dir)

    print('-------------------------------------------------------------------------------------------------------')
    print('Starting checking the files in {0}...'.format(dataset_type_dir))

    image_dir = os.path.join(dataset_type_dir, 'images')
    label_dir = os.path.join(dataset_type_dir, 'labels')
    uncertainty_map_dir = os.path.join(dataset_type_dir, 'uncertainty-maps') if load_uncertainty_map else ''

    assert os.path.isdir(image_dir)
    assert os.path.isdir(label_dir)
    if load_uncertainty_map:
        assert os.path.isdir(uncertainty_map_dir)

    filename_list = os.listdir(image_dir)

    for filename in filename_list:
        # each file's extension must be 'png'
        assert filename.split('.')[-1] == 'png'

        assert os.path.exists(os.path.join(image_dir, filename))
        assert os.path.exists(os.path.join(label_dir, filename))

        if load_uncertainty_map:
            assert os.path.exists(os.path.join(uncertainty_map_dir, filename.replace('png', 'nii')))

    print('Checking passed: all of the involved {} files are legal with extension.'.format(len(filename_list)))
    print('-------------------------------------------------------------------------------------------------------')

    return filename_list


class MicroCalcificationDataset(Dataset):
    def __init__(self, data_root_dir, mode, enable_random_sampling, pos_to_neg_ratio, image_channels, cropping_size,
                 dilation_radius, load_uncertainty_map, calculate_micro_calcification_number, enable_data_augmentation,
                 enable_vertical_flip=False, enable_horizontal_flip=False):

        super(MicroCalcificationDataset, self).__init__()

        # the data root path must exist
        assert os.path.isdir(data_root_dir)

        # the mode must be one of 'training', 'validation' or 'test'
        assert mode in ['training', 'validation', 'test']
        self.mode = mode

        # enable_random_sampling must be a bool variable
        assert isinstance(enable_random_sampling, bool)
        self.enable_random_sampling = enable_random_sampling

        # pos_to_neg_ratio must be a positive number
        assert pos_to_neg_ratio > 0
        self.pos_to_neg_ratio = pos_to_neg_ratio

        # image_channels must be a positive number
        assert image_channels > 0
        self.image_channels = image_channels

        # cropping_size contains length of height and width
        assert len(cropping_size) == 2
        self.cropping_size = cropping_size

        # dilation_radius must be a non-negative integer
        assert dilation_radius >= 0
        assert dilation_radius == int(dilation_radius)
        self.dilation_radius = dilation_radius

        # load_uncertainty_map must be a bool variable
        assert isinstance(load_uncertainty_map, bool)
        self.load_uncertainty_map = load_uncertainty_map

        # calculate_micro_calcification_number must be a bool variable
        assert isinstance(calculate_micro_calcification_number, bool)
        self.calculate_micro_calcification_number = calculate_micro_calcification_number

        # enable_data_augmentation is a bool variable
        assert isinstance(enable_data_augmentation, bool)
        self.enable_data_augmentation = enable_data_augmentation

        # enable_vertical_flip is a bool variable
        assert isinstance(enable_vertical_flip, bool)
        self.enable_vertical_flip = enable_vertical_flip

        # enable_horizontal_flip is a bool variable
        assert isinstance(enable_horizontal_flip, bool)
        self.enable_horizontal_flip = enable_horizontal_flip

        # the image directory of positive and negative patches respectively
        self.positive_patch_dataset_type_dir = os.path.join(data_root_dir, 'positive_patches', self.mode)
        self.negative_patch_dataset_type_dir = os.path.join(data_root_dir, 'negative_patches', self.mode)

        # the image filename list of positive and negative patches respectively
        self.positive_patch_filename_list = generate_and_check_filename_list(self.positive_patch_dataset_type_dir)
        self.negative_patch_filename_list = generate_and_check_filename_list(self.negative_patch_dataset_type_dir)

        # the mixed image filename list including the positive and negative patches
        self.mixed_patch_filename_list = self.positive_patch_filename_list + self.negative_patch_filename_list
        self.mixed_patch_filename_list.sort()

        return

    def __getitem__(self, index):
        """
        :param index
        :return: Tensors
        """
        if self.enable_random_sampling:
            # calculate probability threshold for the random number according the specified pos_to_neg_ratio
            prob_threshold = self.pos_to_neg_ratio / (self.pos_to_neg_ratio + 1)

            # sample positive or negative patch randomly
            if random.random() >= prob_threshold:
                filename = random.sample(self.positive_patch_filename_list, 1)[0]
                image_path = os.path.join(self.positive_patch_dataset_type_dir, 'images', filename)
                image_level_label = [1]
            else:
                filename = random.sample(self.negative_patch_filename_list, 1)[0]
                image_path = os.path.join(self.negative_patch_dataset_type_dir, 'images', filename)
                image_level_label = [0]
        else:
            # sample an image file from mixed_patch_filename_list
            filename = self.mixed_patch_filename_list[index]

            # firstly assume this is a positive patch
            image_path = os.path.join(self.positive_patch_dataset_type_dir, 'images', filename)
            image_level_label = [1]

            # turn it to a negative one, if positive patch directory doesn't contain this image
            if not os.path.exists(image_path):
                image_path = os.path.join(self.negative_patch_dataset_type_dir, 'images', filename)
                image_level_label = [0]

        # get the corresponding pixel-level label path
        pixel_level_label_path = image_path.replace('images', 'labels')

        # get the corresponding uncertainty-map path
        uncertainty_map_path = image_path.replace('images', 'uncertainty-maps')
        uncertainty_map_path = uncertainty_map_path.replace('.png', '.nii')

        # check the existence of the sampled patch
        assert os.path.exists(image_path)
        assert os.path.exists(pixel_level_label_path)
        if self.load_uncertainty_map:
            assert os.path.exists(uncertainty_map_path)

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
        #
        # process pixel-level label                            # normal tissue: 0 (.png) -> 0 (tensor)
        pixel_level_label_np[pixel_level_label_np == 255] = 1  # micro calcification: 255 (.png) -> 1 (tensor)
        pixel_level_label_np[pixel_level_label_np == 165] = 0  # other lesion: 165 (.png) -> 0 (tensor)
        pixel_level_label_np[pixel_level_label_np == 85] = 0  # background: 85 (.png) -> 0 (tensor)

        # load uncertainty maps
        uncertainty_map_np = np.zeros_like(image_np)
        if self.load_uncertainty_map:
            uncertainty_map_image = sitk.ReadImage(uncertainty_map_path, sitk.sitkFloat32)
            uncertainty_map_np = sitk.GetArrayFromImage(uncertainty_map_image)
            uncertainty_map_np = uncertainty_map_np.squeeze()

        # check the consistency of size between image, its pixel-level label and uncertainty-map (fake / authentic)
        assert image_np.shape == pixel_level_label_np.shape == uncertainty_map_np.shape

        # implement data augmentation only when the variable enable_data_augmentation is set True
        if self.enable_data_augmentation:
            if self.enable_vertical_flip and random.random() >= 0.5:
                image_np = np.flipud(image_np)
                pixel_level_label_np = np.flipud(pixel_level_label_np)
                uncertainty_map_np = np.flipud(uncertainty_map_np)
            if self.enable_horizontal_flip and random.random() >= 0.5:
                image_np = np.fliplr(image_np)
                pixel_level_label_np = np.fliplr(pixel_level_label_np)
                uncertainty_map_np = np.fliplr(uncertainty_map_np)

        # guarantee image_np, pixel_level_label_np and uncertainty_map keep contiguous after data augmentation
        image_np = np.ascontiguousarray(image_np)
        pixel_level_label_np = np.ascontiguousarray(pixel_level_label_np)
        uncertainty_map_np = np.ascontiguousarray(uncertainty_map_np)

        # dilate pixel-level label only when the variable dilation_radius is a positive integer
        if self.dilation_radius > 0:
            pixel_level_label_dilated_np = dilate_image_level_label(pixel_level_label_np, self.dilation_radius)
        else:
            pixel_level_label_dilated_np = pixel_level_label_np

        # calculate the number of the annotated micro calcifications
        if self.calculate_micro_calcification_number:
            # when it is a positive patch
            if image_level_label[0] == 1:
                # generate the connected component matrix
                connected_components = measure.label(pixel_level_label_np, connectivity=2)
                micro_calcification_number = [connected_components.max()]
            # when it is a negative patch
            else:
                micro_calcification_number = [0]
        else:
            micro_calcification_number = [-1]

        # convert ndarray to tensor
        #
        # image tensor
        image_tensor = torch.FloatTensor(image_np).unsqueeze(dim=0)  # shape: [C, H, W]
        #
        # pixel-level label tensor
        pixel_level_label_tensor = torch.LongTensor(pixel_level_label_np)  # shape: [H, W]
        #
        # dilated pixel-level label tensor
        pixel_level_label_dilated_tensor = torch.LongTensor(pixel_level_label_dilated_np)  # shape: [H, W]
        #
        # uncertainty-map tensor
        uncertainty_map_tensor = torch.FloatTensor(uncertainty_map_np)  # shape: [H, W]
        #
        # image-level label tensor
        image_level_label_tensor = torch.LongTensor(image_level_label)  # shape: [1]
        #
        # micro calcification number label tensor
        micro_calcification_number_label_tensor = torch.LongTensor(micro_calcification_number)  # shape: [1]

        # resize images, masks and uncertainty maps if the actual size is not consistent with the target size
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

            pixel_level_label_dilated_tensor = pixel_level_label_dilated_tensor.float()
            pixel_level_label_dilated_tensor = torch.nn.functional.interpolate(
                pixel_level_label_dilated_tensor.unsqueeze(dim=0).unsqueeze(dim=0),
                size=(self.cropping_size[0], self.cropping_size[1]),
                scale_factor=None, mode='nearest').squeeze().long()

            uncertainty_map_tensor = torch.nn.functional.interpolate(
                uncertainty_map_tensor.unsqueeze(dim=0).unsqueeze(dim=0),
                size=(self.cropping_size[0], self.cropping_size[1]),
                scale_factor=None, mode='bilinear', align_corners=False).squeeze()

        assert len(image_tensor.shape) == 3
        assert len(pixel_level_label_tensor.shape) == 2
        assert len(pixel_level_label_dilated_tensor.shape) == 2
        assert pixel_level_label_tensor.shape == pixel_level_label_dilated_tensor.shape == uncertainty_map_tensor.shape
        assert len(image_level_label_tensor.shape) == 1
        assert len(micro_calcification_number_label_tensor.shape) == 1
        assert image_tensor.shape[0] == self.image_channels
        assert image_tensor.shape[1] == pixel_level_label_tensor.shape[0] == self.cropping_size[0]
        assert image_tensor.shape[2] == pixel_level_label_tensor.shape[1] == self.cropping_size[1]

        return image_tensor, pixel_level_label_tensor, pixel_level_label_dilated_tensor, uncertainty_map_tensor, \
               image_level_label_tensor, micro_calcification_number_label_tensor, filename

    def __len__(self):

        return len(self.mixed_patch_filename_list)
