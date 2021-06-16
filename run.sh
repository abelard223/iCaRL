source activate pytorch
#nohup python3 main.py --dataset CIFAR100 --model alexnet --gpu_id 0 &> alexnet_cifar100.txt &
nohup python3 main.py --dataset CIFAR10 --model alexnet --gpu_id 1 &> alexnet_cifar10.txt &
#nohup python3 main.py --dataset CIFAR100 --model resnet --gpu_id 2 &> resnet_cifar100.txt &
nohup python3 main.py --dataset CIFAR10 --model resnet --gpu_id 1 &> resnet_cifar10.txt &
