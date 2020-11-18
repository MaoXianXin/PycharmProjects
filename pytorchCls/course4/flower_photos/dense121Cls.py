import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import random
import numpy as np
import os
import copy

# 设置哪块显卡可见
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机数种子，使结果可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

# 数据读取
train_batch = 64
test_batch = 64
EPOCH = 200
trainset = torchvision.datasets.ImageFolder('/home/mao/PycharmProjects/pytorchCls/data/flowersDataset/train',
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
                                                transforms.RandomRotation(50),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5059, 0.4448, 0.3112), (0.2443, 0.2193, 0.2216))
                                            ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.ImageFolder('/home/mao/PycharmProjects/pytorchCls/data/flowersDataset/val',
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5088, 0.4513, 0.3220), (0.2478, 0.2197, 0.2264))
                                           ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                         shuffle=True, num_workers=0)
classes = trainloader.dataset.classes

# 对训练集的一个batch图片进行展示
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(train_batch)))

# 计算Acc
def CalcAcc(net, dataloader):
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the dataset: %.4f %%' % (
            100.0 * correct / total))
    return 100.0 * correct / total

# 网络结构定义
import torch.nn as nn
# 我们把网络更改成densenet121
net = models.densenet121(pretrained=True)
num_ftrs = net.classifier.in_features
net.classifier = nn.Linear(in_features=num_ftrs, out_features=len(classes), bias=True)
# net.load_state_dict(torch.load('dense121Cls.pth'))
net.to(device)

# 定义优化器和损失函数
import torch.optim as optim
# import torch_optimizer as optim
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdaBelief(
#     net.parameters(),
#     lr= 1e-1,
#     betas=(0.9, 0.999),
#     eps=1e-3,
#     weight_decay=0,
#     amsgrad=False,
#     weight_decouple=False,
#     fixed_decay=False,
#     rectify=False,
# )
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.5)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)

# 开始模型训练
best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0
for epoch in range(EPOCH):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)  # 分别把图片和标签传送到GPU设备上

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % len(trainloader) == len(trainloader) - 1:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / len(trainloader)))
            running_loss = 0.0
    CalcAcc(net, trainloader)
    testAcc = CalcAcc(net, testloader)
    if testAcc > best_acc:
        print("------saving best model------")
        best_acc = testAcc
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, 'dense121Cls.pth')
print('Finished Training')
net.load_state_dict(torch.load('dense121Cls.pth'))
CalcAcc(net, testloader)

# 进行模型预测
net.eval()  # 预测的时候要加上这句话
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%10s' % classes[labels[j]] for j in range(test_batch)))


outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted:   ', ' '.join('%10s' % classes[predicted[j]]
                                for j in range(test_batch)))

# 计算测试集上每个类别的Acc
# class_correct = list(0. for i in range(len(classes)))
# class_total = list(0. for i in range(len(classes)))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(test_batch):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
#
# for i in range(len(classes)):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))
