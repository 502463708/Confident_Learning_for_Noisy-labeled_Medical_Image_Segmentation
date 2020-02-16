import argparse
import os
import torch
from torch.autograd import Variable
from net.vnet2d_v4 import VNet2d

kMega = 1e6


def ParseArguments():
    parser = argparse.ArgumentParser()

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
                        default=1,
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
    assert args.in_channels > 0
    assert args.out_channels > 0

    model = VNet2d(num_in_channels=args.in_channels, num_out_channels=args.out_channels)
    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / kMega))

    in_images = torch.zeros([args.batch_size, args.in_channels, args.dim_y, args.dim_x])
    in_images = in_images.cuda()
    in_images_v = Variable(in_images, requires_grad=False)

    reconstructions, residues = model(in_images_v)
    assert reconstructions.size()[0] == args.batch_size
    assert reconstructions.size()[1] == args.out_channels

    print("input shape = ", in_images.shape)
    print("reconstructions shape = ", reconstructions.shape)
    print("residue shape = ", residues.shape)
    print("min value of reconstructions = ", reconstructions.min())
    print("max value of reconstructions = ", reconstructions.max())


# debug only
# outputs = outputs.squeeze().cpu().detach().numpy()
# print(outputs)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = ParseArguments()

    while True:
        TestVdnet2dOutputChannels(args)

# TestVdnet2dOutputChannels(10, 1, 512, 512)
