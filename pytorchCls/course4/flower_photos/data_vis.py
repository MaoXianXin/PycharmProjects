import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pytorchCls.utils.utils import setup_seed

# 设置随机数种子，使结果可复现
setup_seed(20)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


train_batch = 32
trainset = torchvision.datasets.ImageFolder('/home/mao/Downloads/datasets/flowerDatasets/train',
                                            transform=transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(15),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                     (0.229, 0.224, 0.225))
                                            ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                          shuffle=True, num_workers=4, drop_last=True)
test_batch = 32
testset = torchvision.datasets.ImageFolder('/home/mao/Downloads/datasets/flowerDatasets/val',
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225))
                                           ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                         shuffle=True, num_workers=4, drop_last=True)
classes = trainloader.dataset.classes

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(train_batch)))
