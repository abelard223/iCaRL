import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from myNetwork import network


class iCaRLmodel:

	def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate,
				 dataset, device):

		super(iCaRLmodel, self).__init__()
		self.device = device

		self.imsize = (256, 256)  # (32,32)

		self.epochs = epochs
		self.learning_rate = learning_rate
		self.model = network(numclass, feature_extractor)
		print(self.model)
		self.exemplar_set = []  # [[examplees...]classes...]
		self.class_mean_set = []
		self.numclass = numclass  # total `t` classes
		self.transform = transforms.Compose([  # transforms.Resize(img_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
		])
		self.old_model = None

		self.train_transform = transforms.Compose([
			# transforms.RandomCrop(self.imsize, padding=4),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.ColorJitter(brightness=0.24705882352941178),
			transforms.Resize(self.imsize),
			transforms.ToTensor(),
			transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
		])

		self.test_transform = transforms.Compose([
			transforms.Resize(self.imsize),
			transforms.ToTensor(),
			transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
		])

		self.classify_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(p=1.),
			transforms.Resize(self.imsize),
			transforms.ToTensor(),
			transforms.Normalize((0.5071, 0.4867, 0.4408),
								 (0.2675, 0.2565, 0.2761))
		])

		self.train_dataset = dataset('dataset', transform=self.train_transform, download=True)
		self.test_dataset = dataset('dataset', test_transform=self.test_transform, train=False, download=True)

		self.batchsize = batch_size
		self.memory_size = memory_size
		self.task_size = task_size  # new `t-s` classes, and the original classes

		self.train_loader = None
		self.test_loader = None

	def get_one_hot(self, target, num_class):
		one_hot = torch.zeros(target.shape[0], num_class).to(self.device)
		one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
		# one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1))
		return one_hot

	# get incremental train data
	# incremental
	def before_train(self):
		"""
		1. get data_loaders
		2. modify the model's classifier if the numclass > task_size
		Returns:

		"""
		self.model.eval()
		classes = [self.numclass - self.task_size, self.numclass]
		self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
		if self.numclass > self.task_size:  # due to the original classifier's output dim is task_size, if numclass > task_size, it will always increment the output dim of classifier
			self.model.Incremental_learning(self.numclass)
		self.model.train()
		self.model.to(self.device)

	def _get_train_and_test_dataloader(self, classes):
		self.train_dataset.getTrainData(classes, self.exemplar_set)
		self.test_dataset.getTestData(classes)
		train_loader = DataLoader(dataset=self.train_dataset,
								  shuffle=True,
								  batch_size=self.batchsize)

		test_loader = DataLoader(dataset=self.test_dataset,
								 shuffle=True,
								 batch_size=self.batchsize)

		return train_loader, test_loader

	# train model
	# compute loss
	# evaluate model
	@property
	def train(self):
		accuracy = 0
		opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
		for epoch in range(self.epochs):
			if epoch == 48:
				if self.numclass == self.task_size:
					opt = optim.SGD(self.model.parameters(), lr=1.0 / 5, weight_decay=0.00001)
				else:
					for p in opt.param_groups:
						p['lr'] = self.learning_rate / 5
				print(f"change learning rate:{self.learning_rate / 5:.3f}")
			elif epoch == 62:
				if self.numclass > self.task_size:
					for p in opt.param_groups:
						p['lr'] = self.learning_rate / 25
				else:
					opt = optim.SGD(self.model.parameters(), lr=1.0 / 25, weight_decay=0.00001)
				print("change learning rate:%.3f" % (self.learning_rate / 25))
			elif epoch == 80:
				if self.numclass == self.task_size:
					opt = optim.SGD(self.model.parameters(), lr=1.0 / 125, weight_decay=0.00001)
				else:
					for p in opt.param_groups:
						p['lr'] = self.learning_rate / 125
				# opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
				print(f"change learning rate:{self.learning_rate / 100:.3f}")
			for step, (indexs, images, target) in enumerate(self.train_loader):
				images, target = images.to(self.device), target.to(self.device)
				loss_value = self._compute_loss(indexs, images, target)
				opt.zero_grad()
				loss_value.backward()
				opt.step()
				print(f'epoch:{epoch:d},step:{step:d},loss:{loss_value.item():.3f}')
			accuracy = self._test(self.test_loader, 1)
			print(f'epoch:{epoch:d},accuracy:{accuracy:.3f}')
		return accuracy

	def _test(self, testloader, mode):
		if mode == 0:
			print("compute NMS")
		self.model.eval()
		correct, total = 0, 0
		for setp, (indexs, imgs, labels) in enumerate(testloader):
			imgs, labels = imgs.to(self.device), labels.to(self.device)
			with torch.no_grad():
				outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
			predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
			correct += (predicts.cpu() == labels.cpu()).sum()
			total += len(labels)
		accuracy = 100 * correct / total
		self.model.train()
		return accuracy

	def _compute_loss(self, indexs, imgs, target):
		output = self.model(imgs)
		target = self.get_one_hot(target, self.numclass)
		output, target = output.to(self.device), target.to(self.device)
		if self.old_model == None:
			return F.binary_cross_entropy_with_logits(output, target)
		else:
			# old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
			old_target = torch.sigmoid(self.old_model(imgs))
			old_task_size = old_target.shape[1]
			target[..., :old_task_size] = old_target
			return F.binary_cross_entropy_with_logits(output, target)

	# change the size of examplar
	def after_train(self, accuracy):
		"""

		Args:
			accuracy:

		Returns:

		"""
		self.model.eval()
		m = int(self.memory_size / self.numclass)
		self._reduce_exemplar_sets(m)
		for i in range(self.numclass - self.task_size, self.numclass):
			print('construct class %s examplar:' % (i), end='')
			images = self.train_dataset.get_image_class(i)
			self._construct_exemplar_set(images, m)
		self.numclass += self.task_size
		self.compute_exemplar_class_mean()
		self.model.train()
		KNN_accuracy = self._test(self.test_loader, 0)
		print(f'NMS accuracy：{str(KNN_accuracy.item())}')
		filename = f'model/accuracy:{accuracy:.3f}_KNN_accuracy:{KNN_accuracy:.3f}_increment:{i + 10:d}_net.pkl'
		torch.save(self.model, filename)
		self.old_model = torch.load(filename)
		self.old_model.to(self.device)
		self.old_model.eval()

	def _construct_exemplar_set(self, images, m):
		class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
		exemplar = []
		now_class_mean = np.zeros((1, 512))

		for i in range(m):
			# shape：batch_size*512
			x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
			# shape：batch_size
			x = np.linalg.norm(x, axis=1)
			index = np.argmin(x)
			now_class_mean += feature_extractor_output[index]
			exemplar.append(images[index])

		print("the size of exemplar :%s" % (str(len(exemplar))))
		self.exemplar_set.append(exemplar)

	# self.exemplar_set.append(images)

	def _reduce_exemplar_sets(self, m):
		"""
			reduce exemplar sets for class 1~(s-1)
		Args:
			m:
		"""
		for index in range(len(self.exemplar_set)):
			self.exemplar_set[index] = self.exemplar_set[index][:m]
			print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

	def Image_transform(self, images, transform):
		data = transform(Image.fromarray(images[0])).unsqueeze(0)
		for index in range(1, len(images)):
			data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
		return data

	def compute_class_mean(self, images, transform):
		x = self.Image_transform(images, transform).to(self.device)
		feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
		# feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
		class_mean = np.mean(feature_extractor_output, axis=0)
		return class_mean, feature_extractor_output

	def compute_exemplar_class_mean(self):
		self.class_mean_set = []
		for index, exemplar in enumerate(self.exemplar_set):
			print("compute the class mean of %s" % (str(index)))
			class_mean, _ = self.compute_class_mean(exemplar, self.transform)
			class_mean_, _ = self.compute_class_mean(exemplar, self.classify_transform)
			class_mean = (class_mean / np.linalg.norm(class_mean) + class_mean_ / np.linalg.norm(class_mean_)) / 2
			self.class_mean_set.append(class_mean)

	def classify(self, test):
		result = []
		test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
		# test = self.model.feature_extractor(test).detach().cpu().numpy()
		class_mean_set = np.array(self.class_mean_set)
		for target in test:
			x = target - class_mean_set
			x = np.linalg.norm(x, ord=2, axis=1)
			x = np.argmin(x)
			result.append(x)
		return torch.tensor(result)
