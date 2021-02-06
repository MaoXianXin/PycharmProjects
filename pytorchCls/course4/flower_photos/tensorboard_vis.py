# 导入函数库
import torch
from torch.utils.tensorboard import SummaryWriter  # 用于记录训练日志
from torchsummary import summary
from pytorchCls.models.resnet import resnet18, resnet50
from pytorchCls.models.vgg import vgg16, vgg19
from pytorchCls.models.densenet import densenet121

# 声明一个writer实例，用于写入events文件
writer = SummaryWriter('./runs/network_visualization')

# 声明一个网络实例
model_ft = densenet121()

# 模拟输入，要和预训练模型的shape对应上
inputs = torch.ones([1, 3, 224, 224], dtype=torch.float32)

# 把模型写到硬盘上
writer.add_graph(model_ft, inputs)
writer.close()

if __name__ == '__main__':
    model = model_ft.cuda(0)
    # summary(model, input_size=(3, 224, 224))