import argparse
import os
import time
import torch

from loss.slsr_loss import SLSRLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_test',
                        type=int,
                        default=100,
                        help='number of test')

    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='number of patches in each batch')

    parser.add_argument('--num_classes',
                        type=int,
                        default=1,
                        help='1 for grayscale, 3 for RGB images')

    parser.add_argument('--height',
                        type=int,
                        default=5,
                        help='height of images')

    parser.add_argument('--width',
                        type=int,
                        default=5,
                        help='width of images')

    args = parser.parse_args()

    return args


def TestSLSRLoss(args):
    loss_func = SLSRLoss()

    for i in range(args.num_test):
        start_time = time.time()
        preds = torch.rand(args.batch_size, 2, args.height, args.width).cuda()

        labels = torch.rand(args.batch_size, args.height, args.width).cuda()
        labels[labels <= 0.5] = 0
        labels[labels > 0.5] = 1
        labels = labels.cuda()

        confident_maps = torch.rand(args.batch_size, args.height, args.width)
        confident_maps[confident_maps <= 0.5] = 0
        confident_maps[confident_maps > 0.5] = 1
        confident_maps = confident_maps.cuda()

        loss = loss_func(preds, labels, confident_maps)

        print('time:', time.time() - start_time, 'loss: ', loss.item())
        print('\n')

    return


if __name__ == '__main__':
    args = ParseArguments()
    TestSLSRLoss(args)
