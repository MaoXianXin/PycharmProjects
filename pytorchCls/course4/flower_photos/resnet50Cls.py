import torch
from pytorchCls.utils.eval import calc_acc
from pytorchCls.utils.utils import setup_seed, get_dataloader, show_img, predict_batch, set_gpu
from pytorchCls.utils.model import define_model, define_optim, start_train

# 设置哪块显卡可见
device = set_gpu()

# 设置随机数种子，使结果可复现
setup_seed(20)

# 数据读取
train_batch = 64
test_batch = 2
EPOCH = 200
trainloader, testloader, classes = get_dataloader(train_batch, test_batch)

# 对训练集的一个batch图片进行展示
show_img(trainloader, classes, batch_size=train_batch)

# 网络结构定义
net = define_model(classes)
net.to(device)

# 定义优化器和损失函数
criterion, optimizer = define_optim(net)

# 开始模型训练
start_train(net, EPOCH, trainloader, device, optimizer, criterion, testloader, classes, test_batch)

# 训练后在测试集上进行评测
net.load_state_dict(torch.load('resnet50Cls.pth'))
calc_acc(net, testloader, device)

# 进行模型预测
predict_batch(net, testloader, classes, test_batch, device)
