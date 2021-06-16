#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:jbx 
@license: Apache Licence 
@file: AlexNet.py 
@time: 2021/06/14
@email: jianbingxiaman@gmail.com
@description:
"""
from torch import nn
from torchvision.models import alexnet


class AlexNet(nn.Module):
	def __init__(self, num_classes, pretrained=False):
		super(AlexNet, self).__init__()
		self.base_net = alexnet(pretrained=pretrained)
		self.shared_cnn_layers = self.base_net.features
		self.adap_avg_pool = self.base_net.avgpool
		self.shared_fc_layers = self.base_net.classifier[:6]
		self.in_features = self.base_net.classifier[6].in_features


		self.fc = nn.Linear(self.in_features, num_classes) # not forward

	def forward(self, x):
		cnn_out = self.shared_cnn_layers(x)
		cnn_out = self.adap_avg_pool(cnn_out)
		cnn_out_flatten = cnn_out.contiguous().view(cnn_out.size(0), -1)
		shared_fc_out = self.shared_fc_layers(cnn_out_flatten)
		return shared_fc_out
