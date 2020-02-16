"""
This network is derived from vnet2d_v1.
New feature: add sigmoid right before output.
"""
import torch
import torch.nn as nn


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


class InputBlock(nn.Module):
	""" input block of 2d vnet """

	def __init__(self, in_channels, out_channels):
		super(InputBlock, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm2d(out_channels)
		self.act = nn.ReLU(inplace=True)

	def forward(self, input):
		out = self.act(self.bn(self.conv(input)))
		return out


class ConvBnRelu2(nn.Module):
	""" classic combination: conv + batch normalization [+ relu] """

	def __init__(self, in_channels, out_channels, ksize, padding, do_act=True):
		super(ConvBnRelu2, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=padding, groups=1)
		self.bn = nn.BatchNorm2d(out_channels)
		self.do_act = do_act
		if do_act:
			self.act = nn.ReLU(inplace=True)

	def forward(self, input):
		out = self.bn(self.conv(input))
		if self.do_act:
			out = self.act(out)
		return out


class BottConvBnRelu2(nn.Module):
	"""Bottle neck structure"""

	def __init__(self, channels, ratio, do_act=True):
		super(2, self).__init__()
		self.conv1 = ConvBnRelu2(channels, channels / ratio, ksize=1, padding=0, do_act=True)
		self.conv2 = ConvBnRelu2(channels / ratio, channels / ratio, ksize=3, padding=1, do_act=True)
		self.conv3 = ConvBnRelu2(channels / ratio, channels, ksize=1, padding=0, do_act=do_act)

	def forward(self, input):
		out = self.conv3(self.conv2(self.conv1(input)))
		return out


class ResidualBlock2(nn.Module):
	""" 2d residual block with variable number of convolutions """

	def __init__(self, channels, ksize, padding, num_convs):
		super(ResidualBlock2, self).__init__()

		layers = []
		for i in range(num_convs):
			if i != num_convs - 1:
				layers.append(ConvBnRelu2(channels, channels, ksize, padding, do_act=True))
			else:
				layers.append(ConvBnRelu2(channels, channels, ksize, padding, do_act=False))

		self.ops = nn.Sequential(*layers)
		self.act = nn.ReLU(inplace=True)

	def forward(self, input):

		output = self.ops(input)
		return self.act(input + output)


class BottResidualBlock2(nn.Module):
	""" block with bottle neck conv"""

	def __init__(self, channels, ratio, num_convs):
		super(BottResidualBlock2, self).__init__()
		layers = []
		for i in range(num_convs):
			if i != num_convs - 1:
				layers.append(BottConvBnRelu2(channels, ratio, True))
			else:
				layers.append(BottConvBnRelu2(channels, ratio, False))

		self.ops = nn.Sequential(*layers)
		self.act = nn.ReLU(inplace=True)

	def forward(self, input):
		output = self.ops(input)
		return self.act(input + output)


class DownBlock(nn.Module):
	""" downsample block of 2d v-net """

	def __init__(self, in_channels, num_convs, use_bottle_neck=False):
		super(DownBlock, self).__init__()
		out_channels = in_channels * 2
		self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
		self.down_bn = nn.BatchNorm2d(out_channels)
		self.down_act = nn.ReLU(inplace=True)
		if use_bottle_neck:
			self.rblock = BottResidualBlock2(out_channels, 4, num_convs)
		else:
			self.rblock = ResidualBlock2(out_channels, 3, 1, num_convs)

	def forward(self, input):
		out = self.down_act(self.down_bn(self.down_conv(input)))
		out = self.rblock(out)
		return out


class UpBlock(nn.Module):
	""" Upsample block of 2d v-net """

	def __init__(self, in_channels, out_channels, num_convs, use_bottle_neck=False):
		super(UpBlock, self).__init__()
		self.up_conv = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2, groups=1)
		self.up_bn = nn.BatchNorm2d(out_channels // 2)
		self.up_act = nn.ReLU(inplace=True)
		if use_bottle_neck:
			self.rblock = BottResidualBlock2(out_channels, 4, num_convs)
		else:
			self.rblock = ResidualBlock2(out_channels, 3, 1, num_convs)

	def forward(self, input, skip):
		out = self.up_act(self.up_bn(self.up_conv(input)))
		out = torch.cat((out, skip), 1)
		out = self.rblock(out)
		return out


class OutputBlock(nn.Module):
	""" output block of 2d v-net """

	def __init__(self, in_channels, num_classes):
		super(OutputBlock, self).__init__()
		self.num_classes = num_classes

		out_channels = num_classes
		self.class_conv1 = nn.Conv2d(
			in_channels, out_channels, kernel_size=3, padding=1)
		self.class_bn1 = nn.BatchNorm2d(out_channels)
		self.class_act1 = nn.ReLU(inplace=True)
		self.class_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
		# self.softmax = nn.Softmax(1)

	def forward(self, input):
		out = self.class_act1(self.class_bn1(self.class_conv1(input)))
		out = self.class_conv2(out)
		out = torch.sigmoid(out)
		return out


class VNet2d(nn.Module):
	""" 2d v-net """

	def __init__(self, num_in_channels, num_out_channels):
		super(VNet2d, self).__init__()
		self.in_block = InputBlock(num_in_channels, 16)
		self.down_32 = DownBlock(16, 1, use_bottle_neck=False)
		self.down_64 = DownBlock(32, 2, use_bottle_neck=False)
		self.down_128 = DownBlock(64, 3, use_bottle_neck=False)
		self.up_128 = UpBlock(128, 128, 3, use_bottle_neck=False)
		self.up_64 = UpBlock(128, 64, 2, use_bottle_neck=False)
		self.up_32 = UpBlock(64, 32, 1, use_bottle_neck=False)
		self.out_block = OutputBlock(32, num_out_channels)

	def forward(self, input_tensor):
		out16 = self.in_block(input_tensor)
		out32 = self.down_32(out16)
		out64 = self.down_64(out32)
		out128 = self.down_128(out64)
		out = self.up_128(out128, out64)
		out = self.up_64(out, out32)
		out = self.up_32(out, out16)
		output_tensor = self.out_block(out)

		assert input_tensor.shape == output_tensor.shape
		residue_tensor = torch.abs(input_tensor - output_tensor)

		return output_tensor, residue_tensor

	def max_stride(self):
		return 16
