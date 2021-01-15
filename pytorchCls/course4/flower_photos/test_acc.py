from pytorchCls.utils.eval import calc_acc
from pytorchCls.utils.utils import setup_seed, set_gpu
from pytorchCls.utils.model import define_model, define_optim
import torch
import torchvision
import torchvision.transforms as transforms

# 设置哪块显卡可见
device = set_gpu()

# 设置随机数种子，使结果可复现
setup_seed(20)

# 数据读取
test_batch = 96
testset = torchvision.datasets.ImageFolder('/home/mao/Downloads/datasets/flowerDatasets/test',
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               # transforms.Normalize((0.485, 0.456, 0.406),
                                               #                      (0.229, 0.224, 0.225))
                                           ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                         shuffle=True, num_workers=4, drop_last=False)
classes = testloader.dataset.classes

# 网络结构定义
net = define_model(classes)
net.to(device)

# 定义优化器和损失函数
criterion, optimizer = define_optim(net)

# 训练后在测试集上进行评测
net.load_state_dict(torch.load('resnet50Cls.pth'))
calc_acc(net, testloader, device)
