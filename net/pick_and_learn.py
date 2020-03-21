import torch
import torch.nn as nn

from net.vggnet import vgg16_bn
from net.vnet2d_v3 import VNet2d


def kaiming_weight_init(m, bn_std=0.02):
    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        version_tokens = torch.__version__.split('.')
        if int(version_tokens[0]) == 0 and int(version_tokens[1]) < 4:
            nn.init.kaiming_normal(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def ApplyKaimingInit(net):
    net.apply(kaiming_weight_init)


class PLNet2d(nn.Module):
    """ 2d pick-and-learn-net """

    def __init__(self, num_in_channels, num_out_channels, alpha=2):
        super(PLNet2d, self).__init__()

        self.segmentation_module = VNet2d(num_in_channels, num_out_channels)
        self.quality_awareness_module = vgg16_bn(in_channels=num_in_channels + 1, alpha=alpha)

        return

    def forward(self, images_tensor, masks_tensor, use_softmax=False):
        concated_tensor = torch.cat([images_tensor, masks_tensor.unsqueeze(1).float()], 1)

        out_seg = self.segmentation_module(images_tensor, use_softmax)
        out_qam = self.quality_awareness_module(concated_tensor)

        return out_seg, out_qam

    def get_name(self):
        return 'PLNet2d'
