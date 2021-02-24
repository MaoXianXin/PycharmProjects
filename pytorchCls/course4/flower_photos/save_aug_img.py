import torch
import torchvision
import torchvision.transforms as transforms
import os
from pytorchCls.utils.utils import setup_seed

# 设置随机数种子，使结果可复现
setup_seed(20)

train_batch = 32
trainset = torchvision.datasets.ImageFolder('/home/mao/Downloads/datasets/flowerDatasets/train',
                                            transform=transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(15),
                                                transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)
                                            ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                          shuffle=False, num_workers=4, drop_last=True)
classes = trainloader.dataset.classes

for index, data in enumerate(trainset):
    print(index, data[1])
    img_path = '/home/mao/Downloads/datasets/flowerDatasets/train/'+classes[data[1]]
    if not os.path.exists(img_path):
        os.makedirs(img_path, exist_ok=True)
    data[0].save('/home/mao/Downloads/datasets/flowerDatasets/train/'+classes[data[1]]+'/'+str(index)+'.jpg')