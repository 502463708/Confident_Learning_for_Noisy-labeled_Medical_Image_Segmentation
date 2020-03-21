import argparse
import os
import time
import torch

from loss.weighted_cross_entropy_loss import WeightedCrossEntropyLoss

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

    parser.add_argument('--in_channels',
                        type=int,
                        default=1,
                        help='1 for grayscale, 3 for RGB images')

    parser.add_argument('--num_classes',
                        type=int,
                        default=2,
                        help='predicted class number')

    parser.add_argument('--height',
                        type=int,
                        default=256,
                        help='height of images')

    parser.add_argument('--width',
                        type=int,
                        default=256,
                        help='width of images')

    args = parser.parse_args()

    return args


def TestWeightedCrossEntropyLoss(args):
    loss_func = WeightedCrossEntropyLoss()

    for i in range(args.num_test):
        start_time = time.time()
        preds = torch.rand(args.batch_size, 2, args.height, args.width).cuda()

        labels = torch.randint(0, args.num_classes, (args.batch_size, args.height, args.width)).cuda()

        weights = torch.rand(args.batch_size, 1).cuda()

        loss = loss_func(preds, labels, weights)

        print('time:', time.time() - start_time, 'loss: ', loss.item())
        print('\n')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestWeightedCrossEntropyLoss(args)
