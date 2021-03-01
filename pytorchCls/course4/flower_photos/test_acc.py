from pytorchCls.utils.eval import calc_acc
from pytorchCls.utils.utils import setup_seed, set_gpu
from pytorchCls.utils.model import define_model
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 设置哪块显卡可见
device = set_gpu('0')

# 设置随机数种子，使结果可复现
setup_seed(20)

# 数据读取
test_batch = 96
testset = torchvision.datasets.ImageFolder('/home/mao/Downloads/datasets/natural-scenes/seg_test',
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225))
                                           ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                         shuffle=True, num_workers=4, drop_last=False)
classes = testloader.dataset.classes

# 网络结构定义
net = define_model(classes)
net.to(device)


# 训练后在测试集上进行评测
net.load_state_dict(torch.load('resnet18Cls-19.pth'))
calc_acc(net, testloader, device)


predictions = []
groundtruth = []
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.to('cpu'))
        groundtruth.extend(np.asarray(labels.to('cpu')))
    print(classification_report(groundtruth, predictions, target_names=classes))

confusionMatrix = confusion_matrix(groundtruth, predictions)

print(pd.DataFrame(confusionMatrix, index=classes, columns=classes))