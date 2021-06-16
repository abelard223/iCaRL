# PyTorch Implementation of  iCaRL



A PyTorch Implementation of [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725).



## requirement

python3.6

Pytorch1.3.0 linux

numpy

PIL

## Properties

### model

- alexnet
- resnet for ResNet18

### dataset

- cifar100 for 10 tasks
- cifar10 for 5 tasks

## run

```shell
sh run.sh
```

```shell
nohup python3 main.py --dataset CIFAR100 --model alexnet --gpu_id 0 &> alexnet_cifar100.txt &
nohup python3 main.py --dataset CIFAR10 --model alexnet --gpu_id 1 &> alexnet_cifar10.txt &
nohup python3 main.py --dataset CIFAR100 --model resnet --gpu_id 2 &> resnet_cifar100.txt &
nohup python3 main.py --dataset CIFAR10 --model resnet --gpu_id 1 &> resnet_cifar10.txt &
```



