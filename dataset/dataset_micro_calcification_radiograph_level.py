import cv2
import numpy as np
import os
import torch

from torch.utils.data import Dataset


class MicroCalcificationRadiographLevelDataset(Dataset):
    def __init__(self, data_root_dir, mode, image_channels=1):

        super(MicroCalcificationRadiographLevelDataset, self).__init__()

        # the data root path must exist
        assert os.path.isdir(data_root_dir)

        # the mode must be one of 'training', 'validation' or 'test'
        assert mode in ['training', 'validation', 'test']
        self.mode = mode

        # the image directory of radiographs
        self.radiograph_image_dir = os.path.join(data_root_dir, self.mode, 'images')

        # the image filename list of radiographs
        self.radiograph_filename_list = self.generate_and_check_filename_list(self.radiograph_image_dir)
        self.radiograph_filename_list.sort()

        # image_channels must be a positive number
        assert image_channels > 0
        self.image_channels = image_channels

        return

    def __getitem__(self, index):
        """
        :param index
        :return: Tensor
        """
        # sample an image file from mixed_patch_filename_list
        filename = self.radiograph_filename_list[index]

        # firstly assume this is a positive patch
        image_path = os.path.join(self.radiograph_image_dir, filename)

        # get the corresponding pixel-level label path
        pixel_level_label_path = image_path.replace('images', 'labels')

        # check the existence of the sampled patch
        assert os.path.exists(image_path)
        assert os.path.exists(pixel_level_label_path)

        # load image
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_np = image_np.astype(np.float)

        # normalize the intensity range of image: [0, 255] -> [0, 1]
        image_np /= 255.0

        # load pixel-level label
        pixel_level_label_np = cv2.imread(pixel_level_label_path, cv2.IMREAD_GRAYSCALE)
        pixel_level_label_np = pixel_level_label_np.astype(np.uint8)

        # process pixel-level label                            # normal tissue: 0 (.png) -> 0 (tensor)
        pixel_level_label_np[pixel_level_label_np == 255] = 1  # micro calcification: 255 (.png) -> 1 (tensor)
        pixel_level_label_np[pixel_level_label_np == 165] = 2  # other lesion: 165 (.png) -> 2 (tensor)
        pixel_level_label_np[pixel_level_label_np == 85] = 3   # background: 85 (.png) -> 3 (tensor)

        # generate radiograph-level label
        radiograph_level_label = [0]
        if np.where(pixel_level_label_np == 3)[0].shape[0] + np.where(pixel_level_label_np == 0)[0].shape[0] < \
                pixel_level_label_np.size:
            radiograph_level_label = [1]

        # check the consistency of size between image and its pixel-level label
        assert image_np.shape == pixel_level_label_np.shape

        # check the content of pixel-level-label
        assert pixel_level_label_np.max() <= 3

        # convert ndarray to tensor
        #
        # image tensor
        image_tensor = torch.FloatTensor(image_np).unsqueeze(dim=0)  # shape: [C, H, W]
        #
        # pixel-level label tensor
        pixel_level_label_tensor = torch.LongTensor(pixel_level_label_np)  # shape: [H, W]
        #
        # image-level label tensor
        radiograph_level_label_tensor = torch.LongTensor(radiograph_level_label)  # shape: [1]

        assert len(image_tensor.shape) == 3
        assert len(pixel_level_label_tensor.shape) == 2
        assert image_tensor.shape[0] == self.image_channels
        assert image_tensor.shape[1] == pixel_level_label_tensor.shape[0]
        assert image_tensor.shape[2] == pixel_level_label_tensor.shape[1]

        return image_tensor, pixel_level_label_tensor, radiograph_level_label_tensor, filename

    def __len__(self):

        return len(self.radiograph_filename_list)

    def generate_and_check_filename_list(self, absolute_dir):
        # check the correctness of the given dir
        assert os.path.isdir(absolute_dir)

        print('-------------------------------------------------------------------------------------------------------')
        print('Starting checking the files in {0}...'.format(absolute_dir))

        filename_list = os.listdir(absolute_dir)
        for filename in filename_list:
            # each file's extension must be 'png'
            assert filename.split('.')[-1] == 'png'

        print('Checking passed: all of the involved {} files are legal with extension.'.format(len(filename_list)))
        print('-------------------------------------------------------------------------------------------------------')

        return filename_list
