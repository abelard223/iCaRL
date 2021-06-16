import argparse

import torch

from AlexNet import AlexNet
from ResNet import ResNet18
from iCIFAR10 import iCIFAR10
from iCIFAR100 import iCIFAR100
from iCaRL import iCaRLmodel


def print_options(parser, opt):
	"""Print and save options

	It will print both current options and default values(if different).
	It will save options into a text file / [checkpoints_dir] / opt.txt
	"""
	message = ''
	message += '----------------- Options ---------------\n'
	for k, v in sorted(vars(opt).items()):
		comment = ''
		default = parser.get_default(k)
		if v != default:
			comment = '\t[default: %s]' % str(default)
		message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
	message += '----------------- End -------------------'
	print(message)


def main():
	if opt.dataset == 'CIFAR100':
		total_numclasses = 100
		dataset = iCIFAR100
		numclass = 10
		task_size = 10
	else:
		total_numclasses = 10
		dataset = iCIFAR10
		numclass = 2
		task_size = 2

	device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
	if opt.model == 'resnet':
		feature_extractor = ResNet18(pretrained=False, num_classes=total_numclasses)
	elif opt.model == 'alexnet':
		feature_extractor = AlexNet(pretrained=False, num_classes=total_numclasses)
	else:
		raise ValueError(f'Expected model name in [alexnet,resnet], but got {opt.model}')

	epochs = 100

	batch_size = 16
	memory_size = 2000
	learning_rate = 2.0
	model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate,
					   dataset=dataset, device=device)
	# model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))
	num_task = total_numclasses // task_size
	for _ in range(num_task):
		model.before_train()
		accuracy = model.train
		model.after_train(accuracy)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='iCaRL')
	parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'alexnet'],
						help='the model selected in [resnet,alexnet]')
	parser.add_argument('--dataset', type=str, default='CIFAR100', help='the dataset selected in [cifar10,cifar100]')
	parser.add_argument('--gpu_id', type=str, default='0', help='the gpu id')
	opt, _ = parser.parse_known_args()

	opt.dataset = opt.dataset.upper()
	opt.model = opt.model.lower()

	print_options(parser, opt)
	main()
