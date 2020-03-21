import argparse
import os
import torch

from net.pick_and_learn import PLNet2d

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

    parser.add_argument('--out_classes',
                        type=int,
                        default=2,
                        help='out channels number')

    parser.add_argument('--dim_x',
                        type=int,
                        default=256,
                        help='the pixels of patch height')

    parser.add_argument('--dim_y',
                        type=int,
                        default=256,
                        help='the pixels of patch width')

    args = parser.parse_args()

    return args


def TestPLNet2d(args):
    assert args.test_number > 0
    assert args.batch_size > 0
    assert args.in_channels > 0
    assert args.out_classes > 0

    model = PLNet2d(num_in_channels=args.in_channels, num_out_channels=args.out_classes)
    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / kMega))

    for test_idx in range(args.test_number):
        print("-------------------------------------------------------------------------------------------------------")
        print("Testing {} out of {}".format(test_idx, args.test_number))
        image_tensors = torch.rand([args.batch_size, args.in_channels, args.dim_y, args.dim_x]).cuda()
        label_tensors = torch.rand([args.batch_size, args.dim_y, args.dim_x]).cuda()
        label_tensors[label_tensors <= 0.5] = 0
        label_tensors[label_tensors > 0.5] = 1

        out_seg_tensors, out_qam_tensors = model(image_tensors, label_tensors)

        assert out_seg_tensors.size()[0] == out_qam_tensors.size()[0] == args.batch_size

        print("image shape = ", image_tensors.shape)
        print("label shape = ", label_tensors.shape)
        print("segmentation output shape = ", out_seg_tensors.shape)
        print("qam output shape = ", out_qam_tensors.shape)

        print("min value of segmentation output = ", out_seg_tensors.min())
        print("max value of segmentation output = ", out_seg_tensors.max())

        print("min value of qam output = ", out_qam_tensors.min())
        print("max value of qam output = ", out_qam_tensors.max())

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestPLNet2d(args)
