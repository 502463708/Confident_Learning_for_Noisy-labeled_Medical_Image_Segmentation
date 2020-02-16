"""
This file contains all of the common configuration items involved in
training, validation, and test stages of calcification reconstruction.
"""

from easydict import EasyDict as edict


__C = edict()
cfg = __C

# general parameters
__C.general = {}
__C.general.data_root_dir = '/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/'
__C.general.saving_dir = '/data/lars/models/20191112_5764_uCs_patch_level_reconstruction_singleclassdiceloss_default_dilation_radius_7/'
__C.general.cuda_device_idx = '0'  # specify the index of the gpu devices to be occupied

# dataset parameters
__C.dataset = {}
__C.dataset.image_channels = 1  # this is a single-channel image
__C.dataset.cropping_size = [112, 112]  # [H, W] (pixel)
__C.dataset.enable_random_sampling = True  # True: randomly sample only during training
__C.dataset.pos_to_neg_ratio = 1  # hyper-parameter of randomly sampling
__C.dataset.dilation_radius = 7  # pixel-level label to be dilated, 0 -> will not be dilated
__C.dataset.load_uncertainty_map = False  # indicating whether loading the uncertainty map
__C.dataset.calculate_micro_calcification_number = False  # indicating whether calculate the number of calcification

# data augmentation parameters
__C.dataset.augmentation = {}
__C.dataset.augmentation.enable_data_augmentation = True  # whether implement augmentation during training
__C.dataset.augmentation.enable_vertical_flip = True
__C.dataset.augmentation.enable_horizontal_flip = True

# loss
__C.loss = {}
__C.loss.name = 'UncertaintyTTestLossV3'  # only 'TTestLoss', 'TTestLossV2', 'TTestLossV3', 'TTestLossV4',
# 'SoftTTestLoss', 'SingleClassDiceLoss', 'SingleClassTverskyLoss', 'UncertaintyTTestLossV1' , 'UncertaintyTTestLossV2',
# 'UncertaintyTTestLossV3' is supported now
#
__C.loss.t_test_loss = {}
__C.loss.t_test_loss.beta = 0.8
__C.loss.t_test_loss.lambda_p = 1
__C.loss.t_test_loss.lambda_n = 0.1
#
__C.loss.soft_t_test_loss = {}
__C.loss.soft_t_test_loss.beta = 0.8
__C.loss.soft_t_test_loss.lambda_p = 1
__C.loss.soft_t_test_loss.lambda_n = 0.1
__C.loss.soft_t_test_loss.sp_ratio = 0.01
#
__C.loss.tversky_loss = {}
__C.loss.tversky_loss.alpha = 0.25  # the weight set in tversky loss for focusing on FPs
#
__C.loss.uncertainty_t_test_loss_v1 = {}
__C.loss.uncertainty_t_test_loss_v1.beta = 0.8
__C.loss.uncertainty_t_test_loss_v1.lambda_p = 1
__C.loss.uncertainty_t_test_loss_v1.lambda_n = 0.1
__C.loss.uncertainty_t_test_loss_v1.u_low = 0.02
__C.loss.uncertainty_t_test_loss_v1.u_up = 0.1
__C.loss.uncertainty_t_test_loss_v1.w_low = 0.2
__C.loss.uncertainty_t_test_loss_v1.w_up = 0.8
#
__C.loss.uncertainty_t_test_loss_v2 = {}
__C.loss.uncertainty_t_test_loss_v2.beta = 0.8
__C.loss.uncertainty_t_test_loss_v2.lambda_p = 1
__C.loss.uncertainty_t_test_loss_v2.lambda_n = 0.1
__C.loss.uncertainty_t_test_loss_v2.uncertainty_threshold = 0.02
#
__C.loss.uncertainty_t_test_loss_v3 = {}
__C.loss.uncertainty_t_test_loss_v3.beta = 0.8
__C.loss.uncertainty_t_test_loss_v3.lambda_p = 1
__C.loss.uncertainty_t_test_loss_v3.lambda_n = 0.1
__C.loss.uncertainty_t_test_loss_v3.uncertainty_threshold = 0.02

# net
__C.net = {}
__C.net.name = 'vnet2d_v2'  # name of the .py file implementing network architecture
__C.net.in_channels = 1
__C.net.out_channels = 1

# training parameters
__C.train = {}
__C.train.num_epochs = 1001  # number of training epoch
__C.train.save_epochs = 50  # save ckpt every x epochs
__C.train.start_save_best_ckpt = 50  # start saving best ckpt on validation set from the x-th epoch
__C.train.batch_size = 24
__C.train.num_threads = 8

# learning rate scheduler
__C.lr_scheduler = {}
__C.lr_scheduler.lr = 1e-3  # the initial learning rate
__C.lr_scheduler.step_size = 50  # decay learning rate every x epochs
__C.lr_scheduler.gamma = 0.95  # the learning rate decay

# the metrics related parameters
__C.metrics = {}
__C.metrics.prob_threshold = 0.1  # residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1
__C.metrics.area_threshold = 3.14 * 7 * 7 / 4  # connected components whose area < area_threshold will be discarded
__C.metrics.distance_threshold = 14  # candidates (distance between calcification < distance_threshold) is recalled
__C.metrics.slack_for_recall = True  #

# the visdom related parameters
__C.visdom = {}
__C.visdom.port = 9999  # port of visdom
__C.visdom.update_batches = 5  # update images displayed in visdom every x batch


'''
One NVDIA TITAN XP with 12 Mb memory can bear a training load with batch_size=260 
'''