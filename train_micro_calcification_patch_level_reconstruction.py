import importlib
import numpy as np
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
import visdom

from common.utils import save_best_ckpt
from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from metrics.metrics_patch_level_reconstruction import MetricsReconstruction
from logger.logger import Logger
from loss.single_class_dice_loss import SingleClassDiceLoss
from loss.single_class_tversky_loss import SingleClassTverskyLoss
from loss.soft_t_test_loss import SoftTTestLoss
from loss.t_test_loss import TTestLoss
from loss.t_test_loss_v2 import TTestLossV2
from loss.t_test_loss_v3 import TTestLossV3
from loss.t_test_loss_v4 import TTestLossV4
from loss.uncertainty_t_test_loss_v1 import UncertaintyTTestLossV1
from loss.uncertainty_t_test_loss_v2 import UncertaintyTTestLossV2
from loss.uncertainty_t_test_loss_v3 import UncertaintyTTestLossV3
from torch.utils.data import DataLoader
from time import time

# the environment related global variables are specified here
#
# specify the GPUs to be occupied
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.general.cuda_device_idx
cudnn.benchmark = True


def iterate_for_an_epoch(training, epoch_idx, data_loader, net, loss_func, metrics, visdom_obj, logger=None,
                         optimizer=None):
    # training == True -> training mode: backward gradients
    # training == False -> evaluation mode: do not backward gradients
    assert isinstance(training, bool)
    assert epoch_idx >= 0

    if training:
        assert optimizer is not None
        net = net.train()
        if logger is not None:
            logger.write('--------------------------------------------------------------------------------------------')
            logger.write('start training epoch: {}'.format(epoch_idx))
    else:
        net = net.eval()
        if logger is not None:
            logger.write('--------------------------------------------------------------------------------------------')
            logger.write('start evaluating epoch: {}'.format(epoch_idx))

    # this variable is created for recording loss of each batch
    loss_for_each_batch_list = list()

    # these variable is created for recording the annotated calcifications,
    # recalled calcifications and false positive calcifications
    calcification_num_epoch_level = 0
    recall_num_epoch_level = 0
    FP_num_epoch_level = 0

    # start time of this epoch
    start_time_for_epoch = time()

    # iterating through each batch
    for batch_idx, (images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor,
                    uncertainty_maps_tensor, image_level_labels_tensor, _, _) in enumerate(data_loader):

        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()
        #
        image_level_labels_tensor = image_level_labels_tensor.cuda()

        # network forward
        reconstructed_images_tensor, prediction_residues_tensor = net(images_tensor)

        # calculate loss of this batch
        if loss_func.get_name() == 'TTestLoss':
            loss = loss_func(prediction_residues_tensor, image_level_labels_tensor, logger)
        elif loss_func.get_name() == 'TTestLossV2':
            pixel_level_labels_tensor = pixel_level_labels_tensor.cuda()
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, image_level_labels_tensor, pixel_level_labels_dilated_tensor,
                             logger)
        elif loss_func.get_name() == 'TTestLossV3':
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, pixel_level_labels_dilated_tensor, logger)
        elif loss_func.get_name() == 'TTestLossV4':
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, pixel_level_labels_dilated_tensor, logger)
        elif loss_func.get_name() == 'SoftTTestLoss':
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, pixel_level_labels_dilated_tensor, logger)
        elif loss_func.get_name() == 'SingleClassDiceLoss':
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, pixel_level_labels_dilated_tensor, logger)
        elif loss_func.get_name() == 'SingleClassTverskyLoss':
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, pixel_level_labels_dilated_tensor, logger)
        elif loss_func.get_name() == 'UncertaintyTTestLossV1':
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            uncertainty_maps_tensor = uncertainty_maps_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, pixel_level_labels_dilated_tensor, uncertainty_maps_tensor,
                             logger)
        elif loss_func.get_name() == 'UncertaintyTTestLossV2':
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            uncertainty_maps_tensor = uncertainty_maps_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, pixel_level_labels_dilated_tensor, uncertainty_maps_tensor,
                             logger)
        elif loss_func.get_name() == 'UncertaintyTTestLossV3':
            pixel_level_labels_dilated_tensor = pixel_level_labels_dilated_tensor.cuda()
            uncertainty_maps_tensor = uncertainty_maps_tensor.cuda()
            loss = loss_func(prediction_residues_tensor, pixel_level_labels_dilated_tensor, uncertainty_maps_tensor,
                             logger)

        loss_for_each_batch_list.append(loss.item())

        # backward gradients only when training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics
        post_process_residues_np, calcification_num_batch_level, recall_num_batch_level, FP_num_batch_level, _ = \
            metrics.metric_batch_level(prediction_residues_tensor, pixel_level_labels_tensor)
        calcification_num_epoch_level += calcification_num_batch_level
        recall_num_epoch_level += recall_num_batch_level
        FP_num_epoch_level += FP_num_batch_level

        # print logging information
        if logger is not None:
            logger.write('epoch: {}, batch: {}, loss: {:.4f}, consuming time: {:.4f}s'
                         .format(epoch_idx, batch_idx, loss.item(), time() - start_time_for_batch))

        # update images display in visdom
        if batch_idx % cfg.visdom.update_batches == 0:
            try:
                visdom_obj.images(
                    images_tensor,
                    win='I{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='I{}'.format('T' if training else 'V'))
                )
                visdom_obj.images(
                    reconstructed_images_tensor,
                    win='Recon{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='Recon{}'.format('T' if training else 'V'))
                )
                visdom_obj.images(
                    prediction_residues_tensor,
                    win='Res{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='Res{}'.format('T' if training else 'V'))
                )
                visdom_obj.images(
                    np.expand_dims(post_process_residues_np, axis=1),
                    win='PRes{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='PRes{}'.format('T' if training else 'V'))
                )
                visdom_obj.images(
                    pixel_level_labels_tensor.unsqueeze(dim=1),
                    win='PL{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='PL{}'.format('T' if training else 'V'))
                )
                visdom_obj.images(
                    pixel_level_labels_dilated_tensor.unsqueeze(dim=1),
                    win='PLD{}'.format('T' if training else 'V'),
                    nrow=1,
                    opts=dict(title='PLD{}'.format('T' if training else 'V'))
                )
            except BaseException as err:
                print('Error message: ', err)

    # calculate loss of this epoch
    average_loss_of_this_epoch = np.array(loss_for_each_batch_list).mean()

    # record metric on validation set for determining the best model to be saved
    if not training:
        metrics.determine_saving_metric_on_validation_list.append(recall_num_epoch_level - FP_num_epoch_level)

    if logger is not None:
        logger.write('{} of epoch {} finished'.format('training' if training else 'evaluating', epoch_idx))
        logger.write('epoch: {}, loss: {:.4f}, consuming time: {:.4f}s'.format(epoch_idx,
                                                                               average_loss_of_this_epoch,
                                                                               time() - start_time_for_epoch))
        logger.write('--------------------------------------------------------------------------------------------')

    # update loss of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([average_loss_of_this_epoch]),
                    win='loss',
                    update='append',
                    name='{}_loss'.format('training' if training else 'validation'),
                    opts=dict(title='loss'))

    # update annotated calcification number loss of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([calcification_num_epoch_level]),
                    win='metrics_{}'.format('training' if training else 'validation'),
                    update='append',
                    name='annotated calcifications',
                    opts=dict(title='metrics_{}'.format('training' if training else 'validation')))

    # update recalled calcification number loss of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([recall_num_epoch_level]),
                    win='metrics_{}'.format('training' if training else 'validation'),
                    update='append',
                    name='recalled calcifications',
                    opts=dict(title='metrics_{}'.format('training' if training else 'validation')))

    # update false positive calcification number loss of this epoch in the visdom
    visdom_obj.line(X=np.array([epoch_idx]),
                    Y=np.array([FP_num_epoch_level]),
                    win='metrics_{}'.format('training' if training else 'validation'),
                    update='append',
                    name='FP calcifications',
                    opts=dict(title='metrics_{}'.format('training' if training else 'validation')))

    return


if __name__ == '__main__':
    # create a folder for saving purpose
    ckpt_dir = os.path.join(cfg.general.saving_dir, 'ckpt')
    if not os.path.exists(cfg.general.saving_dir):
        os.makedirs(cfg.general.saving_dir)
        os.makedirs(ckpt_dir)

        # copy related config and net .py file to the saving dir
        shutil.copyfile('./config/config_micro_calcification_patch_level_reconstruction.py',
                        os.path.join(cfg.general.saving_dir,
                                     'config_micro_calcification_patch_level_reconstruction.py'))
        shutil.copyfile('./net/{0}.py'.format(cfg.net.name),
                        os.path.join(cfg.general.saving_dir, '{0}.py'.format(cfg.net.name)))

    # initialize logger
    logger = Logger(cfg.general.saving_dir)

    # import the network package
    try:
        net_package = importlib.import_module('net.{}'.format(cfg.net.name))
    except BaseException:
        print('failed to import package: {}'.format('net.' + cfg.net.name))
    #
    # define the network
    net = net_package.VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)

    # check whether the ckpt dir is empty
    ckpt_file_list = os.listdir(ckpt_dir)
    if len(ckpt_file_list) == 0:
        net = torch.nn.DataParallel(net).cuda()
        net_package.ApplyKaimingInit(net)
        logger.write('Training from scratch...')
    else:
        # find the latest saved ckpt
        latest_ckpt_idx = np.array([int(ckpt_file.split('_')[2].split('.')[0]) for ckpt_file in ckpt_file_list]).max()
        latest_ckpt_file = 'net_epoch_{0}.pth'.format(latest_ckpt_idx)
        checkpoint_path = os.path.join(ckpt_dir, latest_ckpt_file)

        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(checkpoint_path))
        logger.write('Load ckpt: {0}...'.format(latest_ckpt_file))

    # setup metrics
    metrics = MetricsReconstruction(cfg.metrics.prob_threshold, cfg.metrics.area_threshold,
                                    cfg.metrics.distance_threshold, cfg.metrics.slack_for_recall)

    # setup Visualizer
    visdom_display_name = cfg.general.saving_dir.split('/')[-2]
    visdom_obj = visdom.Visdom(env=visdom_display_name, port=cfg.visdom.port)

    # create dataset and data loader for training
    training_dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                                 mode='training',
                                                 enable_random_sampling=cfg.dataset.enable_random_sampling,
                                                 pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                                 image_channels=cfg.dataset.image_channels,
                                                 cropping_size=cfg.dataset.cropping_size,
                                                 dilation_radius=cfg.dataset.dilation_radius,
                                                 load_uncertainty_map=cfg.dataset.load_uncertainty_map,
                                                 calculate_micro_calcification_number=cfg.dataset.calculate_micro_calcification_number,
                                                 enable_data_augmentation=cfg.dataset.augmentation.enable_data_augmentation,
                                                 enable_vertical_flip=cfg.dataset.augmentation.enable_vertical_flip,
                                                 enable_horizontal_flip=cfg.dataset.augmentation.enable_horizontal_flip)

    training_data_loader = DataLoader(training_dataset, batch_size=cfg.train.batch_size,
                                      shuffle=True, num_workers=cfg.train.num_threads)

    # create dataset and data loader for validation
    validation_dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                                   mode='validation',
                                                   enable_random_sampling=False,
                                                   pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                                   image_channels=cfg.dataset.image_channels,
                                                   cropping_size=cfg.dataset.cropping_size,
                                                   dilation_radius=cfg.dataset.dilation_radius,
                                                   load_uncertainty_map=cfg.dataset.load_uncertainty_map,
                                                   calculate_micro_calcification_number=cfg.dataset.calculate_micro_calcification_number,
                                                   enable_data_augmentation=False)

    validation_data_loader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size,
                                        shuffle=True, num_workers=cfg.train.num_threads)

    # define loss function
    assert cfg.loss.name in ['TTestLoss', 'TTestLossV2', 'TTestLossV3', 'TTestLossV4', 'SoftTTestLoss',
                             'SingleClassDiceLoss', 'SingleClassTverskyLoss', 'UncertaintyTTestLossV1',
                             'UncertaintyTTestLossV2', 'UncertaintyTTestLossV3']
    if cfg.loss.name == 'TTestLoss':
        loss_func = TTestLoss(beta=cfg.loss.t_test_loss.beta, lambda_p=cfg.loss.t_test_loss.lambda_p,
                              lambda_n=cfg.loss.t_test_loss.lambda_n)
    elif cfg.loss.name == 'TTestLossV2':
        loss_func = TTestLossV2(beta=cfg.loss.t_test_loss.beta, lambda_p=cfg.loss.t_test_loss.lambda_p,
                                lambda_n=cfg.loss.t_test_loss.lambda_n)
    elif cfg.loss.name == 'TTestLossV3':
        loss_func = TTestLossV3(beta=cfg.loss.t_test_loss.beta, lambda_p=cfg.loss.t_test_loss.lambda_p,
                                lambda_n=cfg.loss.t_test_loss.lambda_n)
    elif cfg.loss.name == 'TTestLossV4':
        loss_func = TTestLossV4(beta=cfg.loss.t_test_loss.beta, lambda_p=cfg.loss.t_test_loss.lambda_p,
                                lambda_n=cfg.loss.t_test_loss.lambda_n)
    elif cfg.loss.name == 'SoftTTestLoss':
        loss_func = SoftTTestLoss(beta=cfg.loss.soft_t_test_loss.beta, lambda_p=cfg.loss.soft_t_test_loss.lambda_p,
                                  lambda_n=cfg.loss.soft_t_test_loss.lambda_n,
                                  sp_ratio=cfg.loss.soft_t_test_loss.sp_ratio)
    elif cfg.loss.name == 'SingleClassDiceLoss':
        loss_func = SingleClassDiceLoss()
    elif cfg.loss.name == 'SingleClassTverskyLoss':
        loss_func = SingleClassTverskyLoss(cfg.loss.tversky_loss.alpha)
    elif cfg.loss.name == 'UncertaintyTTestLossV1':
        loss_func = UncertaintyTTestLossV1(beta=cfg.loss.uncertainty_t_test_loss_v1.beta,
                                           lambda_p=cfg.loss.uncertainty_t_test_loss_v1.lambda_p,
                                           lambda_n=cfg.loss.uncertainty_t_test_loss_v1.lambda_n,
                                           u_low=cfg.loss.uncertainty_t_test_loss_v1.u_low,
                                           u_up=cfg.loss.uncertainty_t_test_loss_v1.u_up,
                                           w_low=cfg.loss.uncertainty_t_test_loss_v1.w_low,
                                           w_up=cfg.loss.uncertainty_t_test_loss_v1.w_up)
    elif cfg.loss.name == 'UncertaintyTTestLossV2':
        loss_func = UncertaintyTTestLossV2(beta=cfg.loss.uncertainty_t_test_loss_v2.beta,
                                           lambda_p=cfg.loss.uncertainty_t_test_loss_v2.lambda_p,
                                           lambda_n=cfg.loss.uncertainty_t_test_loss_v2.lambda_n,
                                           uncertainty_threshold=cfg.loss.uncertainty_t_test_loss_v2.uncertainty_threshold)
    elif cfg.loss.name == 'UncertaintyTTestLossV3':
        loss_func = UncertaintyTTestLossV3(beta=cfg.loss.uncertainty_t_test_loss_v3.beta,
                                           lambda_p=cfg.loss.uncertainty_t_test_loss_v3.lambda_p,
                                           lambda_n=cfg.loss.uncertainty_t_test_loss_v3.lambda_n,
                                           uncertainty_threshold=cfg.loss.uncertainty_t_test_loss_v3.uncertainty_threshold)

    # setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr_scheduler.lr)

    # learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler.step_size,
                                                   gamma=cfg.lr_scheduler.gamma)

    # iterating through each epoch
    for epoch_idx in range(0, cfg.train.num_epochs):
        iterate_for_an_epoch(training=True,
                             epoch_idx=epoch_idx,
                             data_loader=training_data_loader,
                             net=net,
                             loss_func=loss_func,
                             metrics=metrics,
                             visdom_obj=visdom_obj,
                             logger=logger,
                             optimizer=optimizer)

        iterate_for_an_epoch(training=False,
                             epoch_idx=epoch_idx,
                             data_loader=validation_data_loader,
                             net=net,
                             loss_func=loss_func,
                             metrics=metrics,
                             visdom_obj=visdom_obj,
                             logger=logger)
        lr_scheduler.step()

        logger.flush()

        # whether to save this model according to config
        if epoch_idx % cfg.train.save_epochs is 0:
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(epoch_idx)))

        # save this model in case that this is the currently best model on validation set
        if epoch_idx >= cfg.train.start_save_best_ckpt:
            save_best_ckpt(metrics, net, ckpt_dir, epoch_idx)
