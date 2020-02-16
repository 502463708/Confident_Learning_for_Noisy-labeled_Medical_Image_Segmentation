import argparse
import os
import time
import torch

from loss.l1_loss import L1Loss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_test',
                        type=int,
                        default=100,
                        help='number of test')

    parser.add_argument('--batch_size',
                        type=int,
                        default=48,
                        help='number of patches in each batch')

    args = parser.parse_args()

    return args


def TestL1Loss(args):
    for i in range(args.num_test):
        start_time = time.time()

        preds = torch.rand(args.batch_size, 1).cuda()
        labels = torch.rand(args.batch_size, 1).cuda()

        loss = L1Loss()(preds, labels)
        print('time:', time.time() - start_time, 'loss: ', loss.item())

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestL1Loss(args)
