import argparse
import os
import torch

from net.vnet2d_v3 import VNet2d

kMega = 1e6
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_number',
                        type=int,
                        default=10,
                        help='number of test')

    parser.add_argument('--batch_size',
                        type=int,
                        default=12,
                        help='number of patches in each batch')

    parser.add_argument('--in_channels',
                        type=int,
                        default=1,
                        help='1 for grayscale, 3 for RGB images')

    parser.add_argument('--out_channels',
                        type=int,
                        default=2,
                        help='out channels number')

    parser.add_argument('--dim_x',
                        type=int,
                        default=112,
                        help='the pixels of patch height')

    parser.add_argument('--dim_y',
                        type=int,
                        default=112,
                        help='the pixels of patch width')

    args = parser.parse_args()

    return args


def TestVdnet2dOutputChannels(args):
    assert args.test_number > 0
    assert args.batch_size > 0
    assert args.in_channels > 0
    assert args.out_channels > 0

    model = VNet2d(num_in_channels=args.in_channels, num_out_channels=args.out_channels)
    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / kMega))

    for test_idx in range(args.test_number):
        print("-------------------------------------------------------------------------------------------------------")
        print("Testing {} out of {}".format(test_idx, args.test_number))
        in_tensors = torch.rand([args.batch_size, args.in_channels, args.dim_y, args.dim_x])
        in_tensors = in_tensors.cuda()

        out_tensors = model(in_tensors, use_softmax=True)

        assert out_tensors.size()[0] == args.batch_size
        assert out_tensors.size()[1] == args.out_channels

        print("input shape = ", in_tensors.shape)
        print("output shape = ", out_tensors.shape)
        print("min value of output = ", out_tensors.min())
        print("max value of output = ", out_tensors.max())

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestVdnet2dOutputChannels(args)
