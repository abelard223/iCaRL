import torch.nn as nn

__all__ = []

from torchvision.models import resnet18


class ResNet18(nn.Module):

	def __init__(self, num_classes=100, pretrained=False):
		self.inplanes = 64
		super(ResNet18, self).__init__()
		self.base_net = resnet18(pretrained=pretrained)
		self.conv1=self.base_net.conv1
		self.bn1=self.base_net.bn1
		self.relu=self.base_net.relu
		self.maxpool=self.base_net.maxpool
		self.layer1=self.base_net.layer1
		self.layer2=self.base_net.layer2
		self.layer3=self.base_net.layer3
		self.layer4=self.base_net.layer4
		self.avgpool = self.base_net.avgpool
		self.in_features = self.base_net.fc.in_features

		self.fc = nn.Linear(self.in_features, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		out_flatten = x.contiguous().view(x.size(0), -1)

		return out_flatten
