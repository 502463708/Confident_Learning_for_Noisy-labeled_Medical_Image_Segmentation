import torch
import torch.nn as nn
import math


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

    return


def ApplyKaimingInit(net):
    net.apply(kaiming_weight_init)

    return


class convblock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1):
        super(convblock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class DCnet(nn.Module):
    # H_out= H_in+2*padding-dilation*(kernels_size-1)
    # H_out=(H_in+2*padding-dilation*(kernelsize-1)-1)/stride+1
    # W_out=(W_in+2*padding-dilation*(kernelsize-1)-1)/stride+1

    def __init__(self, in_channels=1, num_classes=2):
        super(DCnet, self).__init__()
        self.conv0 = convblock(in_channels, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1 = convblock(32, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2 = convblock(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = convblock(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv4 = convblock(64, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv5 = convblock(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv6 = convblock(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7 = convblock(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv8 = convblock(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv9 = convblock(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv10 = convblock(128, 512, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv11 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, input):
        x = self.conv0(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        out = self.conv11(x)
        out=torch.sigmoid(out)
        assert out.shape == input.shape
        residue = torch.abs(input - out)


        return out, residue

