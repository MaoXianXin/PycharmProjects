import torch
import random
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# 设置随机数种子，使结果可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置哪块显卡可见
def set_gpu():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


# 数据读取
def get_dataloader(train_batch, test_batch):
    trainset = torchvision.datasets.ImageFolder('/home/mao/Downloads/datasets/flowerDatasets/train',
                                                transform=transforms.Compose([
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    # transforms.Normalize((0.485, 0.456, 0.406),
                                                    #                      (0.229, 0.224, 0.225))
                                                ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              shuffle=True, num_workers=4, drop_last=True)
    testset = torchvision.datasets.ImageFolder('/home/mao/Downloads/datasets/flowerDatasets/val',
                                               transform=transforms.Compose([
                                                   transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   # transforms.Normalize((0.485, 0.456, 0.406),
                                                   #                      (0.229, 0.224, 0.225))
                                               ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=True, num_workers=4, drop_last=True)
    classes = trainloader.dataset.classes

    return trainloader, testloader, classes


# functions to show an image
def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 对训练集的一个batch图片进行展示
def show_img(dataloader, classes, batch_size):
    # get some random training images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


# 进行模型预测
def predict_batch(net, dataloader, classes, batch_size, device):
    net.eval()  # 预测的时候要加上这句话
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%10s' % classes[labels[j]] for j in range(batch_size)))

    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('Predicted:   ', ' '.join('%10s' % classes[predicted[j]]
                                    for j in range(batch_size)))
