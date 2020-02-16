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
__C.general.data_root_dir = '/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/'
__C.general.saving_dir = '/data/lars/models/20191231_uCs_image_level_classification_CE_default_debug/'
__C.general.cuda_device_idx = '0'  # specify the index of the gpu devices to be occupied

# dataset parameters
__C.dataset = {}
__C.dataset.image_channels = 1  # this is a single-channel image
__C.dataset.cropping_size = [112, 112]  # [H, W] (pixel)
__C.dataset.enable_random_sampling = True  # True: randomly sample only during training
__C.dataset.pos_to_neg_ratio = 1  # hyper-parameter of randomly sampling
__C.dataset.dilation_radius = 0  # pixel-level label to be dilated, 0 -> will not be dilated
__C.dataset.load_uncertainty_map = False  # indicating whether loading the uncertainty map
__C.dataset.calculate_micro_calcification_number = False  # indicating whether calculate the number of calcification

# data augmentation parameters
__C.dataset.augmentation = {}
__C.dataset.augmentation.enable_data_augmentation = True  # whether implement augmentation during training
__C.dataset.augmentation.enable_vertical_flip = True
__C.dataset.augmentation.enable_horizontal_flip = True

# loss
__C.loss = {}
__C.loss.name = 'CrossEntropyLoss'  # only 'CrossEntropyLoss, UncertaintyCrossEntropyLossV1,
# UncertaintyCrossEntropyLossV2' 'L1Loss' implemented
# cfg.dataset.load_uncertainty_map has to be set True when using uncertainty loss
#
__C.loss.uncertainty_cross_entropy_loss_v1 = {}
__C.loss.uncertainty_cross_entropy_loss_v1.upn = 0.02
__C.loss.uncertainty_cross_entropy_loss_v1.epsilon = 0.2
#
__C.loss.uncertainty_cross_entropy_loss_v2 = {}
__C.loss.uncertainty_cross_entropy_loss_v2.uncertainty_threshold = 0.02

# net
__C.net = {}
__C.net.name = 'resnet18'  # name of the .py file implementing network architecture
__C.net.in_channels = 1
__C.net.num_classes = 2
__C.net.activation = None  # only None, 'softmax', 'sigmoid' is supported

# training parameters
__C.train = {}
__C.train.num_epochs = 501  # number of training epoch
__C.train.save_epochs = 50  # save ckpt every x epochs
__C.train.batch_size = 480
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
One NVIDIA TITAN XP with 12 Mb memory can bear a training load with batch_size=1024 
'''
