import torch


# 计算Acc
def calc_acc(net, dataloader, device):
    net.eval()
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print('-----', correct)
    # print('-----', total)
    print('Accuracy of the network on the dataset: %.4f %%' % (
            100.0 * correct / total))
    return 100.0 * correct / total


# 计算每个类别的召回率
def calc_every_acc(net, dataloader, device, classes, batch_size=2):
    net.eval()
    class_correct = list(0.0 for i in range(len(classes)))
    class_total = list(0.0 for i in range(len(classes)))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    # print('------', class_correct)
    # print('------', class_total)
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

