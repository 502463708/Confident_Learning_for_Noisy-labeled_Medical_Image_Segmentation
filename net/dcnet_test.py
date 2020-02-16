import argparse
import os
import torch
from net.dcnet import DCnet
from torch.autograd import Variable

kMega = 1e6

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_time',
                        type=int,
                        default=10,
                        help='number of test run')

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

                        default=1,
                        help='classes number: for reconstruction is 1')


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


def TestDCnet(args):
    assert args.batch_size > 0
    assert args.num_classes > 0

    model = DCnet(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / kMega))


    input_tensor = torch.rand([args.batch_size, args.in_channels, args.dim_y, args.dim_x])
    input_tensor = input_tensor.cuda()

    output_tensor, residue = model(input_tensor)
    assert output_tensor.size()[0] == args.batch_size
    assert output_tensor.size()[1] == args.num_classes


    print("input shape = ", input_tensor.shape)
    print("output shape = ", output_tensor.shape)


if __name__ == '__main__':

    args = ParseArguments()


    for idx in range(args.run_time):
        TestDCnet(args)
