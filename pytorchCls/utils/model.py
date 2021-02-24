import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import copy
import torch
from pytorchCls.utils.eval import calc_acc, calc_every_acc
from pytorchCls.utils.ohem import NLL_OHEM
from pytorchCls.models.resnet import resnet18, resnet50, resnext50_32x4d, wide_resnet50_2
from pytorchCls.models.vgg import vgg16, vgg19
from pytorchCls.models.densenet import densenet121


# 网络结构定义
def define_model(classes):
    net = resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(in_features=num_ftrs, out_features=len(classes), bias=True)
    return net


# 定义优化器和损失函数
def define_optim(net):
    criterion = NLL_OHEM()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    return criterion, optimizer


def start_train(net, EPOCH, trainloader, device, optimizer, criterion, testloader, classes, test_batch):
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    softmax = torch.nn.LogSoftmax(dim=1)
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)  # 分别把图片和标签传送到GPU设备上

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(softmax(outputs), labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % len(trainloader) == len(trainloader) - 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / len(trainloader)))
                running_loss = 0.0
        calc_acc(net, trainloader, device)
        testAcc = calc_acc(net, testloader, device)
        # calc_every_acc(net, testloader, device, classes, batch_size=test_batch)
        if testAcc > best_acc:
            print("------saving best model------")
            best_acc = testAcc
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(best_model_wts, 'resnet18Cls-'+str(epoch+1)+'.pth')
    print('Finished Training')