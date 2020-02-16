import argparse
import copy
import cv2
import importlib
import numpy as np
import os
import shutil
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn

from common.utils import get_ckpt_path, generate_radiograph_level_reconstructed_and_residue_result, \
    post_process_residue_radiograph, get_net_list, generate_uncertainty_maps
from config.config_micro_calcification_patch_level_classification import cfg as c_cfg
from config.config_micro_calcification_patch_level_reconstruction import cfg as r_cfg
from dataset.dataset_micro_calcification_radiograph_level import MicroCalcificationRadiographLevelDataset
from logger.logger import Logger
from metrics.metrics_radiograph_level_detection import MetricsRadiographLevelDetection
from net.vnet2d_v2 import VNet2d
from skimage import measure
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-radiograph-level-roi-extracted-data-split-dataset/',
                        help='The source data dir.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset to be evaluated (training, validation, test).')
    parser.add_argument('--prediction_saving_dir',
                        type=str,
                        default='/data/lars/results/20191109_5764-uCs-micro_calcification_radiograph_level_detection_results_rec_dilatted_7_cls_pos_2_neg_0.5_areath_0.2_probth_0.1/',
                        help='The predicted results saving dir.')
    parser.add_argument('--reconstruction_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='The reconstruction model saved dir.')
    parser.add_argument('--reconstruction_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--mc_epoch_indexes',
                        type=int,
                        default=[410, 420, 430, 440, 450, 460, 470, 480, 490, 500],
                        help='The epoch ckpt index list for generating uncertainty maps'
                             'set null list [] to switch off.')
    parser.add_argument('--classification_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_pos2neg_0.5_classification_CE_default/',
                        help='The classification model saved dir.')
    parser.add_argument('--classification_net_name',
                        type=str,
                        default='resnet18_v1',
                        help='The file name implementing classification network architecture.')
    parser.add_argument('--classification_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--reconstruction_patch_size',
                        type=tuple,
                        default=(112, 112),
                        help='The height and width of patch.')
    parser.add_argument('--patch_stride',
                        type=int,
                        default=56,
                        help='The patch moving stride from one patch to another.')
    parser.add_argument('--prob_threshold',
                        type=float,
                        default=0.1,
                        help='residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=3.14 * 7 * 7 * 0.2,
                        help='Connected components whose area < area_threshold will be discarded.')
    parser.add_argument('--crop_patch_size',
                        type=tuple,
                        default=(56, 56),
                        help='The height and width of patch.')
    parser.add_argument('--resampled_patch_size',
                        type=tuple,
                        default=(224, 224),
                        help='The height and width of patch.')
    parser.add_argument('--score_threshold_stride',
                        type=float,
                        default=0.05,
                        help='The score threshold stride for calculating recalls and FPs.')
    parser.add_argument('--distance_threshold',
                        type=int,
                        default=14,
                        help='Candidates whose distance between calcification < distance_threshold is a recalled one.')
    parser.add_argument('--slack_for_recall',
                        type=bool,
                        default=True,
                        help='The bool variable for slacking recall metric standard.')
    parser.add_argument('--save_visualization_results',
                        type=bool,
                        default=False,
                        help='The bool variable to determine whether save the visualization results.')

    args = parser.parse_args()

    return args


def generate_coordinate_and_score_list(images_tensor, classification_net, pixel_level_label_np,
                                       raw_residue_radiograph_np, processed_residue_radiograph_np, filename, saving_dir,
                                       crop_patch_size, upsampled_patch_size, net_list, mode='detected'):
    # mode must be either 'detected' or 'annotated'
    assert mode in ['detected', 'annotated']

    if saving_dir is not None:
        # make the related dirs
        patch_level_root_saving_dir = os.path.join(saving_dir, filename[:-4])
        patch_visualization_dir = os.path.join(patch_level_root_saving_dir, mode)
        if not os.path.exists(patch_level_root_saving_dir):
            os.mkdir(patch_level_root_saving_dir)
        os.mkdir(patch_visualization_dir)

    height, width = processed_residue_radiograph_np.shape

    if mode == 'detected':
        # mode: detected -> iterate each connected component on processed_residue_radiograph_np
        mask_np = copy.copy(processed_residue_radiograph_np)
        mask_np[processed_residue_radiograph_np > 0] = 1
    else:
        # mode: annotated -> iterate each connected component on pixel_level_label_np
        mask_np = copy.copy(pixel_level_label_np)
        # remain micro calcifications and normal tissue label only
        mask_np[mask_np > 1] = 0

    # generate information of each connected component
    connected_components = measure.label(mask_np)
    props = measure.regionprops(connected_components)

    # created for saving the coordinates and the detected score for this connected component
    coordinate_list = list()
    score_list = list()

    connected_idx = 0
    if len(props) > 0:
        for prop in props:
            connected_idx += 1

            # generate logical indexes for this connected component
            indexes = connected_components == connected_idx

            # record the centroid of this connected component
            coordinate_list.append(np.array(prop.centroid))

            # generate legal start and end idx for row and column
            centroid_row_idx = prop.centroid[0]
            centroid_column_idx = prop.centroid[1]
            #
            centroid_row_idx = np.clip(
                centroid_row_idx, crop_patch_size[0] / 2, height - crop_patch_size[0] / 2)
            centroid_column_idx = np.clip(
                centroid_column_idx, crop_patch_size[1] / 2, width - crop_patch_size[1] / 2)
            #
            start_row_idx = int(centroid_row_idx - crop_patch_size[0] / 2)
            end_row_idx = int(centroid_row_idx + crop_patch_size[0] / 2)
            start_column_idx = int(centroid_column_idx - crop_patch_size[1] / 2)
            end_column_idx = int(centroid_column_idx + crop_patch_size[1] / 2)

            # crop this patch for model inference
            patch_image_tensor = images_tensor[:, :, start_row_idx:end_row_idx, start_column_idx:end_column_idx]
            upsampled_patch_image_tensor = \
                torch.nn.functional.interpolate(patch_image_tensor, size=(upsampled_patch_size[0],
                                                                          upsampled_patch_size[1]),
                                                scale_factor=None, mode='bilinear', align_corners=False)

            # generate the positive class prediction probability
            classification_preds_tensor = classification_net(upsampled_patch_image_tensor)
            classification_preds_tensor = torch.softmax(classification_preds_tensor, dim=1)
            positive_prob = classification_preds_tensor.cpu().detach().numpy().squeeze()[1]

            # MC dropout
            uncertainty_maps_np = generate_uncertainty_maps(net_list, upsampled_patch_image_tensor)
            uncertainty_map_np = uncertainty_maps_np.squeeze()

            # calculate the mean value of this connected component on the residue
            residue_mean = (processed_residue_radiograph_np[indexes]).mean()

            # calculate and record the score of this connected component
            score = positive_prob * residue_mean
            score_list.append(score)

            if saving_dir is not None:
                # process the visualization results
                image_patch_np = copy.copy(patch_image_tensor.cpu().detach().numpy().squeeze())
                #
                pixel_level_label_patch_np = copy.copy(
                    pixel_level_label_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx])
                #
                raw_residue_patch_np = copy.copy(
                    raw_residue_radiograph_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx])
                #
                processed_residue_patch_np = copy.copy(
                    processed_residue_radiograph_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx])
                #
                stacked_np = np.concatenate((np.expand_dims(image_patch_np, axis=0),
                                             np.expand_dims(
                                                 pixel_level_label_patch_np, axis=0),
                                             np.expand_dims(
                                                 raw_residue_patch_np, axis=0),
                                             np.expand_dims(
                                                 processed_residue_patch_np, axis=0)), axis=0)
                stacked_image = sitk.GetImageFromArray(stacked_np)
                #
                image_patch_np *= 255
                raw_residue_patch_np *= 255
                processed_residue_patch_np *= 255
                uncertainty_map_np *= 4 * 255
                #
                pixel_level_label_patch_np[pixel_level_label_patch_np == 1] = 255
                pixel_level_label_patch_np[pixel_level_label_patch_np == 2] = 165
                pixel_level_label_patch_np[pixel_level_label_patch_np == 3] = 85
                #
                image_patch_np = image_patch_np.astype(np.uint8)
                raw_residue_patch_np = raw_residue_patch_np.astype(np.uint8)
                processed_residue_patch_np = processed_residue_patch_np.astype(np.uint8)
                pixel_level_label_patch_np = pixel_level_label_patch_np.astype(np.uint8)
                uncertainty_map_np = uncertainty_map_np.astype(np.uint8)
                uncertainty_map_np = cv2.applyColorMap(uncertainty_map_np, cv2.COLORMAP_JET)
                #
                prob_saving_image = np.zeros((crop_patch_size[0], crop_patch_size[1], 3), np.uint8)
                mean_residue_saving_image = np.zeros((crop_patch_size[0], crop_patch_size[1], 3), np.uint8)
                score_saving_image = np.zeros((crop_patch_size[0], crop_patch_size[1], 3), np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(prob_saving_image, '{:.4f}'.format(positive_prob), (0, 64), font, 1, (0, 255, 255), 2)
                cv2.putText(mean_residue_saving_image, '{:.4f}'.format(residue_mean), (0, 64), font, 1, (255, 0, 255),
                            2)
                cv2.putText(score_saving_image, '{:.4f}'.format(positive_prob * residue_mean), (0, 64), font, 1,
                            (255, 255, 0), 2)

                # saving
                cv2.imwrite(os.path.join(patch_visualization_dir,
                                         filename.replace('.png', '_patch_{:0>3d}_{}_{}_image.png'.format(connected_idx,
                                                                                                          int(centroid_row_idx),
                                                                                                          int(centroid_column_idx)))),
                            image_patch_np)
                cv2.imwrite(os.path.join(patch_visualization_dir,
                                         filename.replace('.png', '_patch_{:0>3d}_mask.png'.format(connected_idx))),
                            pixel_level_label_patch_np)
                cv2.imwrite(os.path.join(patch_visualization_dir,
                                         filename.replace('.png',
                                                          '_patch_{:0>3d}_raw_residue.png'.format(connected_idx))),
                            raw_residue_patch_np)
                cv2.imwrite(os.path.join(patch_visualization_dir,
                                         filename.replace('.png',
                                                          '_patch_{:0>3d}_processed_residue.png'.format(
                                                              connected_idx))),
                            processed_residue_patch_np)
                cv2.imwrite(os.path.join(patch_visualization_dir,
                                         filename.replace('.png',
                                                          '_patch_{:0>3d}_uncertainty.png'.format(
                                                              connected_idx))),
                            uncertainty_map_np)
                cv2.imwrite(os.path.join(patch_visualization_dir,
                                         filename.replace('.png',
                                                          '_patch_{:0>3d}_positive_prob.png'.format(connected_idx))),
                            prob_saving_image)
                cv2.imwrite(os.path.join(patch_visualization_dir,
                                         filename.replace('.png',
                                                          '_patch_{:0>3d}_mean_residue.png'.format(connected_idx))),
                            mean_residue_saving_image)
                cv2.imwrite(os.path.join(patch_visualization_dir,
                                         filename.replace('.png', '_patch_{:0>3d}_score.png'.format(connected_idx))),
                            score_saving_image)
                sitk.WriteImage(stacked_image,
                                os.path.join(patch_visualization_dir,
                                             filename.replace('.png', '_patch_{:0>3d}.nii'.format(connected_idx))))

    return coordinate_list, score_list


def save_radiograph_level_results(images_tensor, pixel_level_label_np, raw_residue_radiograph_np,
                                  processed_residue_radiograph_np, filename, saving_dir):
    # convert tensor into numpy and squeeze channel dimension
    image_np = images_tensor.cpu().detach().numpy().squeeze()

    assert image_np.shape == pixel_level_label_np.shape == raw_residue_radiograph_np.shape == \
           processed_residue_radiograph_np.shape

    # process the visualization results
    stacked_np = np.concatenate((np.expand_dims(image_np, axis=0),
                                 np.expand_dims(pixel_level_label_np, axis=0),
                                 np.expand_dims(processed_residue_radiograph_np, axis=0),
                                 np.expand_dims(raw_residue_radiograph_np, axis=0)), axis=0)
    stacked_image = sitk.GetImageFromArray(stacked_np)
    #
    image_np *= 255
    raw_residue_radiograph_np *= 255
    processed_residue_radiograph_np *= 255
    #
    # process pixel-level label                            # normal tissue: 0 (.png) <- 0 (tensor)
    # micro calcification: 255 (.png) <- 1 (tensor)
    pixel_level_label_np[pixel_level_label_np == 1] = 255
    # other lesion: 165 (.png) <- 2 (tensor)
    pixel_level_label_np[pixel_level_label_np == 2] = 165
    # background: 85 (.png) <- 3 (tensor)
    pixel_level_label_np[pixel_level_label_np == 3] = 85
    #
    image_np = image_np.astype(np.uint8)
    pixel_level_label_np = pixel_level_label_np.astype(np.uint8)
    raw_residue_radiograph_np = raw_residue_radiograph_np.astype(np.uint8)
    processed_residue_radiograph_np = processed_residue_radiograph_np.astype(np.uint8)

    # saving
    cv2.imwrite(os.path.join(saving_dir, filename.replace('.png', '_patch_image.png')), image_np)
    cv2.imwrite(os.path.join(saving_dir, filename.replace('.png', '_patch_pixel_level_label.png')),
                pixel_level_label_np)
    cv2.imwrite(os.path.join(saving_dir, filename.replace('.png', '_patch_raw_residue.png')),
                raw_residue_radiograph_np)
    cv2.imwrite(os.path.join(saving_dir, filename.replace('.png', '_patch_post_processed_residue.png')),
                processed_residue_radiograph_np)
    sitk.WriteImage(stacked_image, os.path.join(saving_dir, filename.replace('png', 'nii')))

    return


def TestMicroCalcificationRadiographLevelDetection(args):
    # start time of this dataset
    start_time_for_dataset = time()

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(args.prediction_saving_dir):
        shutil.rmtree(args.prediction_saving_dir)
    os.mkdir(args.prediction_saving_dir)

    # create dir for saving visualization results
    patch_level_visualization_saving_dir = None
    if args.save_visualization_results:
        visualization_saving_dir = os.path.join(args.prediction_saving_dir, 'qualitative_results')
        radiograph_level_visualization_saving_dir = os.path.join(visualization_saving_dir, 'radiograph_level')
        patch_level_visualization_saving_dir = os.path.join(visualization_saving_dir, 'patch_level')
        #
        os.mkdir(visualization_saving_dir)
        os.mkdir(radiograph_level_visualization_saving_dir)
        os.mkdir(patch_level_visualization_saving_dir)

    # initialize logger
    logger = Logger(args.prediction_saving_dir, 'quantitative_results.txt')
    logger.write_and_print('Dataset: {}'.format(args.data_root_dir))
    logger.write_and_print('Dataset type: {}'.format(args.dataset_type))
    logger.write_and_print('Reconstruction model saving dir: {}'.format(args.reconstruction_model_saving_dir))
    logger.write_and_print('Reconstruction ckpt index: {}'.format(args.reconstruction_epoch_idx))
    logger.write_and_print('Classification model saving dir: {}'.format(args.classification_model_saving_dir))
    logger.write_and_print('Classification ckpt index: {}'.format(args.classification_epoch_idx))

    # define the reconstruction network
    reconstruction_net = VNet2d(num_in_channels=r_cfg.net.in_channels, num_out_channels=r_cfg.net.out_channels)
    #
    mc_reconstruction_net = copy.deepcopy(reconstruction_net)
    #
    # get the reconstruction absolute ckpt path
    reconstruction_ckpt_path = get_ckpt_path(args.reconstruction_model_saving_dir, args.reconstruction_epoch_idx)
    #
    # load ckpt and transfer net into gpu devices
    reconstruction_net = torch.nn.DataParallel(reconstruction_net).cuda()
    reconstruction_net.load_state_dict(torch.load(reconstruction_ckpt_path))
    reconstruction_net = reconstruction_net.eval()
    #
    logger.write_and_print('Load ckpt: {0}...'.format(reconstruction_ckpt_path))

    # get calculate_uncertainty global variance
    calculate_uncertainty = True if len(args.mc_epoch_indexes) > 0 else False

    # get net list for imitating MC dropout process
    net_list = None
    if calculate_uncertainty:
        net_list = get_net_list(mc_reconstruction_net, os.path.join(args.reconstruction_model_saving_dir, 'ckpt'),
                                args.mc_epoch_indexes, logger)

    # import the network package
    try:
        net_package = importlib.import_module('net.{}'.format(args.classification_net_name))
    except BaseException:
        print('failed to import package: {}'.format('net.' + args.classification_net_name))
    #
    # define the classification network
    classification_net = net_package.ResNet18(in_channels=c_cfg.net.in_channels, num_classes=c_cfg.net.num_classes)
    #
    # get the classification absolute ckpt path
    classification_ckpt_path = get_ckpt_path(args.classification_model_saving_dir, args.classification_epoch_idx)
    #
    # load ckpt and transfer net into gpu devices
    classification_net = torch.nn.DataParallel(classification_net).cuda()
    classification_net.load_state_dict(torch.load(classification_ckpt_path))
    classification_net = classification_net.eval()
    #
    logger.write_and_print('Load ckpt: {0}...'.format(classification_ckpt_path))

    # create dataset
    dataset = MicroCalcificationRadiographLevelDataset(data_root_dir=args.data_root_dir, mode=args.dataset_type)

    # create data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=r_cfg.train.num_threads)

    # set up metrics object
    metrics = MetricsRadiographLevelDetection(args.distance_threshold, args.score_threshold_stride)

    for radiograph_idx, (images_tensor, pixel_level_labels_tensor, _, filenames) in enumerate(data_loader):
        filename = filenames[0]

        # logging
        logger.write_and_print('--------------------------------------------------------------------------------------')
        logger.write_and_print(
            'Start evaluating radiograph {} out of {}: {}...'.format(radiograph_idx + 1, dataset.__len__(), filename))

        # start time of this radiograph
        start_time_for_radiograph = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # transfer the tensor into ndarray format
        pixel_level_label_np = pixel_level_labels_tensor.cpu().numpy().squeeze()

        # generated raw radiograph-level residue
        _, raw_residue_radiograph_np = generate_radiograph_level_reconstructed_and_residue_result(images_tensor,
                                                                                                  reconstruction_net,
                                                                                                  pixel_level_label_np,
                                                                                                  args.reconstruction_patch_size,
                                                                                                  args.patch_stride)

        # post-process the raw radiograph-level residue
        processed_residue_radiograph_np = post_process_residue_radiograph(raw_residue_radiograph_np,
                                                                          pixel_level_label_np,
                                                                          args.prob_threshold,
                                                                          args.area_threshold)

        # generate coordinates and score list for the post-processed radiograph-level residue
        pred_coord_list, pred_score_list = generate_coordinate_and_score_list(images_tensor,
                                                                              classification_net,
                                                                              pixel_level_label_np,
                                                                              raw_residue_radiograph_np,
                                                                              processed_residue_radiograph_np,
                                                                              filename,
                                                                              patch_level_visualization_saving_dir,
                                                                              args.crop_patch_size,
                                                                              args.resampled_patch_size,
                                                                              net_list)

        # generate coordinates list for the mask
        label_coord_list, _ = generate_coordinate_and_score_list(images_tensor,
                                                                 classification_net,
                                                                 pixel_level_label_np,
                                                                 raw_residue_radiograph_np,
                                                                 processed_residue_radiograph_np,
                                                                 filename,
                                                                 patch_level_visualization_saving_dir,
                                                                 args.crop_patch_size,
                                                                 args.resampled_patch_size,
                                                                 net_list,
                                                                 mode='annotated')

        # evaluate based on the above three lists
        if args.slack_for_recall:
            detection_result_record_radiograph_level = metrics.metric_all_score_thresholds(pred_coord_list,
                                                                                           pred_score_list,
                                                                                           label_coord_list,
                                                                                           processed_residue_radiograph_np)
        else:
            detection_result_record_radiograph_level = metrics.metric_all_score_thresholds(pred_coord_list,
                                                                                           pred_score_list,
                                                                                           label_coord_list)
        # save radiograph-level visualization results
        if args.save_visualization_results:
            save_radiograph_level_results(images_tensor, pixel_level_label_np, raw_residue_radiograph_np,
                                          processed_residue_radiograph_np, filename,
                                          radiograph_level_visualization_saving_dir)

        # logging
        # print logging information of this radiograph
        logger.write_and_print(
            'Finish evaluating radiograph: {}, consuming time: {:.4f}s'.format(radiograph_idx + 1,
                                                                               time() - start_time_for_radiograph))
        detection_result_record_radiograph_level.print(logger)
        logger.write_and_print('--------------------------------------------------------------------------------------')
        logger.flush()

    # print logging information of this dataset
    logger.write_and_print(
        '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    logger.write_and_print(
        'Finished evaluating this dataset, consuming time: {:.4f}s'.format(time() - start_time_for_dataset))
    metrics.detection_result_record_dataset_level.print(logger)
    logger.write_and_print(
        '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationRadiographLevelDetection(args)
