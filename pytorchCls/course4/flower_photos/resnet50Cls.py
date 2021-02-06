from pytorchCls.utils.utils import setup_seed, get_dataloader, show_img, set_gpu
from pytorchCls.utils.model import define_model, define_optim, start_train

# 设置哪块显卡可见
device = set_gpu('1')

# 设置随机数种子，使结果可复现
setup_seed(20)

# 数据读取
train_batch = 72
test_batch = 72
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