import argparse
import os
import time
import torch

from loss.cross_entropy_loss import CrossEntropyLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_test',
                        type=int,
                        default=100,
                        help='The time of test.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=24,
                        help='The number of patches in a batch.')

    parser.add_argument('--length',
                        type=int,
                        default=1,
                        help='The number of patches in a batch.')

    parser.add_argument('--num_classes',
                        type=int,
                        default=2,
                        help='The number of category including background.')

    args = parser.parse_args()

    return args


def TestCrossEntropyLoss(args):
    for i in range(args.num_test):
        loss_func = CrossEntropyLoss()

        start_time = time.time()

        preds = torch.rand(args.batch_size, args.num_classes, args.length).cuda()

        labels = torch.rand(args.batch_size, args.length).cuda()
        labels[labels <= 0.5] = 0
        labels[labels > 0.5] = 1
        labels = labels.long()

        loss = loss_func(preds, labels)

        print('time:', time.time() - start_time, 'loss: ', loss.item())
        print('\n')

    return


if __name__ == '__main__':
    args = ParseArguments()
    TestCrossEntropyLoss(args)
