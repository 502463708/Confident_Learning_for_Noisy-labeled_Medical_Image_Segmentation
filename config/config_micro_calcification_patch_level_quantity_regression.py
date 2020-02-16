"""
This file contains all of the common configuration items involved in
training, validation, and test stages of calcification image-level
classification.
"""

from easydict import EasyDict as edict

__C = edict()
cfg = __C

# general parameters
__C.general = {}
__C.general.data_root_dir = '/home/groupprofzli/data1/dwz/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/'
__C.general.saving_dir = '/home/groupprofzli/data1/dwz/data/models/20191026_uCs_quantity_regression_L1/'
__C.general.cuda_device_idx = '0,1'  # specify the index of the gpu devices to be occupied

# dataset parameters
__C.dataset = {}
__C.dataset.image_channels = 1  # this is a single-channel image
__C.dataset.cropping_size = [112, 112]  # [H, W] (pixel)
__C.dataset.enable_random_sampling = True  # True: randomly sample only during training
__C.dataset.pos_to_neg_ratio = 1  # hyper-parameter of randomly sampling
__C.dataset.dilation_radius = 0  # pixel-level label to be dilated, 0 -> will not be dilated
__C.dataset.load_uncertainty_map = False  # indicating whether loading the uncertainty map
__C.dataset.calculate_micro_calcification_number = True  # indicating whether calculate the number of calcification

# data augmentation parameters
__C.dataset.augmentation = {}
__C.dataset.augmentation.enable_data_augmentation = True  # whether implement augmentation during training
__C.dataset.augmentation.enable_vertical_flip = True
__C.dataset.augmentation.enable_horizontal_flip = True

# loss
__C.loss = {}
__C.loss.name = 'L1Loss'  # only 'LOneLoss' implemented

# net
__C.net = {}
__C.net.name = 'resnet18'  # name of the .py file implementing network architecture
__C.net.in_channels = 1
__C.net.num_classes = 1

# training parameters
__C.train = {}
__C.train.num_epochs = 501  # number of training epoch
__C.train.save_epochs = 50  # save ckpt every x epochs
__C.train.batch_size = 800
__C.train.num_threads = 8

# learning rate scheduler
__C.lr_scheduler = {}
__C.lr_scheduler.lr = 1e-3  # the initial learning rate
__C.lr_scheduler.step_size = 50  # decay learning rate every x epochs
__C.lr_scheduler.gamma = 0.95  # the learning rate decay

# the metrics related parameters
__C.metrics = {}

# the visdom related parameters
__C.visdom = {}
__C.visdom.port = 9999  # port of visdom
__C.visdom.update_batches = 5  # update images displayed in visdom every x batch

'''
One NVDIA TITAN XP with 12 Mb memory can bear a training load with batch_size=1024 
'''
