from pytorchCls.utils.utils import setup_seed, set_gpu
from pytorchCls.utils.model import define_model
import torch
import torchvision
import torchvision.transforms as transforms
from shutil import copyfile
import os

# 设置哪块显卡可见
device = set_gpu('0')

# 设置随机数种子，使结果可复现
setup_seed(20)

# 数据读取
test_batch = 32
testset = torchvision.datasets.ImageFolder('/home/mao/Downloads/datasets/natural-scenes/seg_train',
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

count = 0
dest_parent_path = '/home/mao/Downloads/datasets/natural-scenesError/'
net.eval()
with torch.no_grad():
    for index, data in enumerate(testset):
        output = torch.argmax(net(torch.unsqueeze(data[0], dim=0).to(device)), -1).to('cpu')
        if data[1] != output.item():
            print(testset.imgs[index], output.item())
            count += 1
            if not os.path.exists(dest_parent_path+classes[testset.imgs[index][1]]):
                os.makedirs(dest_parent_path+classes[testset.imgs[index][1]], exist_ok=True)
            copyfile(testset.imgs[index][0], dest_parent_path+classes[testset.imgs[index][1]]+'/'+testset.imgs[index][0].split('/')[-1])