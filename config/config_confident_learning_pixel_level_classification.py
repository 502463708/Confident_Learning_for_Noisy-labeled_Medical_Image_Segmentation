"""
This file contains all of the common configuration items involved in
training, validation, and test stages of micro calcification pixel-level
classification.
"""

from easydict import EasyDict as edict

__C = edict()
cfg = __C

# general parameters
__C.general = {}
__C.general.data_root_dir = '/data1/minqing/data/JRST/noisy-data-alpha-0.5-beta1-7-beta2-12/all/'
__C.general.saving_dir = '/data1/minqing/models/20200306_JRST_dataset_noisy_alpha-0.5_beta1_7_beta2_12_all_segmentation_lung_slsr_0.8/'
__C.general.cuda_device_idx = '0'  # specify the index of the gpu devices to be occupied

# dataset parameters
__C.dataset = {}
__C.dataset.class_name = 'lung'  # 'clavicle', 'heart', 'lung'
__C.dataset.image_channels = 1  # this is a single-channel image
__C.dataset.cropping_size = [256, 256]  # [H, W] (pixel)
__C.dataset.load_confident_map = True  # True: load confident maps only during training
__C.dataset.enable_random_sampling = True  # True: randomly sampling only during training

# data augmentation parameters
__C.dataset.augmentation = {}
__C.dataset.augmentation.enable_data_augmentation = False  # whether implement augmentation during training
__C.dataset.augmentation.enable_vertical_flip = False
__C.dataset.augmentation.enable_horizontal_flip = True

# loss
__C.loss = {}
__C.loss.name = 'SLSRLoss'  # only 'CrossEntropyLoss', 'SLSRLoss' is supported now

__C.loss.slsrloss = {}
__C.loss.slsrloss.epsilon = 0.8

# net
__C.net = {}
__C.net.name = 'vnet2d_v3'  # name of the .py file implementing network architecture
__C.net.in_channels = 1
__C.net.out_channels = 2

# training parameters
__C.train = {}
__C.train.num_epochs = 1001  # number of training epoch
__C.train.save_epochs = 50  # save ckpt every x epochs
__C.train.batch_size = 4
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
One NVDIA TITAN XP with 12 Mb memory can bear a training load with batch_size=260 
'''
