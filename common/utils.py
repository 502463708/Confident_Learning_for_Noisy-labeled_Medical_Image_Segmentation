import copy
import numpy as np
import os
import torch


def get_ckpt_path(model_saving_dir, epoch_idx=-1):
    """
    Given a dir (where the model is saved) and an index (which ckpt is specified),
    This function returns the absolute ckpt path
    :param model_saving_dir:
    :param epoch_idx:
        default mode: epoch_idx = -1 -> return the best ckpt
        specified mode: epoch_idx >= 0 -> return the specified ckpt
    :return: absolute ckpt path
    """
    assert os.path.exists(model_saving_dir)
    assert epoch_idx >= -1

    ckpt_dir = os.path.join(model_saving_dir, 'ckpt')
    # specified mode: epoch_idx is specified -> load the specified ckpt
    if epoch_idx >= 0:
        ckpt_path = os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(epoch_idx))
    # default mode: epoch_idx is not specified -> load the best ckpt
    else:
        saved_ckpt_list = os.listdir(ckpt_dir)
        best_ckpt_filename = [best_ckpt_filename for best_ckpt_filename in saved_ckpt_list if
                              'net_best_on_validation_set' in best_ckpt_filename][0]
        ckpt_path = os.path.join(ckpt_dir, best_ckpt_filename)

    return ckpt_path


def save_best_ckpt(metrics, net, ckpt_dir, epoch_idx):
    """
    This function can discriminatively save this ckpt in case that it is the currently best ckpt on validation set
    :param metrics:
    :param net:
    :param ckpt_dir:
    :param epoch_idx:
    :return:
    """

    is_best_ckpt = metrics.determine_saving_metric_on_validation_list[-1] == max(
        metrics.determine_saving_metric_on_validation_list)
    if is_best_ckpt:
        # firstly remove the last saved best ckpt
        saved_ckpt_list = os.listdir(ckpt_dir)
        for saved_ckpt_filename in saved_ckpt_list:
            if 'net_best_on_validation_set' in saved_ckpt_filename:
                os.remove(os.path.join(ckpt_dir, saved_ckpt_filename))

        # save the current best ckpt
        torch.save(net.state_dict(),
                   os.path.join(ckpt_dir, 'net_best_on_validation_set_epoch_{}.pth'.format(epoch_idx)))

    return


def extract_classification_preds_channel(preds, channel_idx, use_softmax=True, keep_dim=True):
    """

    :param preds:
    :param channel_idx:
    :param use_softmax:
    :param keep_dim:
    :return:
    """
    assert torch.is_tensor(preds)
    assert len(preds.shape) == 4
    assert channel_idx >= 0

    if use_softmax:
        preds = torch.softmax(preds, dim=1)

    extracted_preds = preds[:, channel_idx, :, :]

    if keep_dim:
        extracted_preds = extracted_preds.unsqueeze(dim=1)
        assert len(preds.shape) == len(extracted_preds.shape)

    return extracted_preds


def get_net_list(network, ckpt_dir, mc_epoch_indexes, logger):
    assert len(mc_epoch_indexes) > 0

    net_list = list()

    for mc_epoch_idx in mc_epoch_indexes:
        net = copy.deepcopy(network)
        ckpt_path = os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(mc_epoch_idx))
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(ckpt_path))
        net = net.eval()
        net_list.append(net)

        logger.write_and_print('Load ckpt: {0} for MC dropout...'.format(ckpt_path))

    return net_list
