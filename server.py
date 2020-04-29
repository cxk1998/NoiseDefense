import torch
import torch.nn as nn
from LeNet import LeNet
import torchvision as tv
import torchvision.transforms as transforms
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
import platform
import math
import random
import numpy as np
import scipy.stats as sts
from sklearn.metrics.pairwise import cosine_similarity
# 超参数设置
epochs = 15  # 遍历数据集次数
BATCH_SIZE = 60  # 批处理尺寸(batch_size)
LR = 0.01  # 学习率
Num_client = 50 # client 数目

# 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# 加载数据集
transform = transforms.ToTensor()  # 定义数据预处理方式
# 判断系统平台
def is_windowssystem():
    return 'Windows' in platform.system()
def is_linuxsystem():
    return 'Linux' in platform.system()

if is_windowssystem():
    MNIST_data = "./dataset"  # windows
if is_linuxsystem():
    MNIST_data = "/home/yjdu/federatedlearning_DP_torch/dataset"  # linux

# 定义训练数据集
trainset = tv.datasets.MNIST(
    root=MNIST_data,
    train=True,
    download=False,
    transform=transform)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# 定义测试数据集
testset = tv.datasets.MNIST(
    root=MNIST_data,
    train=False,
    download=False,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# 分割训练集
dataset_list = list(trainloader)
dataset_len = len(dataset_list)
client_len = dataset_len // Num_client

# for i, data in enumerate(trainloader):
#     inputs, labels = data
# 
#     inputs, labels = inputs.to(device), labels.to(device)

# 网络参数初始化
def weight_init(m):
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # torch.manual_seed(7)   # 随机种子，是否每次做相同初始化赋值
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
    # m中的 weight，bias 其实都是 Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

net = LeNet()
# 初始化网络参数
net.apply(weight_init)  # apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上
# # 提取网络参数
# net_dic = net.state_dict()
# 定义损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer_server = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

clients_net = []
# 分配用户参数 send_back()
for c in range(Num_client):
    t = LeNet()
    clients_net.append(t)

# client_0_net.load_state_dict(net_dic)
# outputs_c0 = client_0_net(dataset_c0)
# loss_c0 = criterion(outputs_c0, client_0_labels)
# loss_c0.backward()


# client训练，获取梯度
def get_client_grad(client_inputs, client_labels, net_dict, client_net, epoch):
    client_net.load_state_dict(net_dict)
    client_outputs = client_net(client_inputs)
    client_loss = criterion(client_outputs, client_labels)
    client_optimizer = optim.SGD(client_net.parameters(), lr=LR/(epoch+1), momentum=0.9)
    client_optimizer.zero_grad()  # 梯度置零 
    client_loss.backward()  # 求取梯度
    # 提取梯度
    client_grad_dict = dict()  # name: params_grad
    params_modules = list(client_net.named_parameters())
    for params_module in params_modules:
        (name, params) = params_module
        params_grad = copy.deepcopy(params.grad)
        client_grad_dict[name] = params_grad
    client_optimizer.zero_grad()  # 梯度置零 
    return client_grad_dict, client_loss


losses = []
accs = []
hv = [[] for i in range(epochs)]
for epoch in range(epochs):
    sum_loss = 0.0
    judge = random.random()
    # 处理数据
    for index in range(client_len):
        clients_grad_dict = []
        for c in range(Num_client):
            client_inputs, client_labels = dataset_list[index + client_len * c]
            client_inputs, client_labels = client_inputs.to(device), client_labels.to(device)
            net_dict = net.state_dict()  # 提取server网络参数
            t, tl = get_client_grad(client_inputs, client_labels, net_dict, clients_net[c], epoch)
            clients_grad_dict.append(t)
            sum_loss += tl
        # 取各client参数梯度均值
        client_average_grad_dict = dict()
        for key in clients_grad_dict[0]:
            temp = []
            noise0 = torch.from_numpy(sts.laplace.rvs(loc=0, scale=0.375, size=clients_grad_dict[0][key].shape)).type(
                torch.float).to(torch.device("cpu"))
            temp.append(clients_grad_dict[0][key]+noise0)
            t_vector = [np.concatenate((np.array((clients_grad_dict[0][key]+noise0).view(1, -1))), axis=0)]
            client_average_grad_dict[key] = (clients_grad_dict[0][key]+noise0) / Num_client
            for c in range(1, Num_client):
                noise = torch.from_numpy(sts.laplace.rvs(loc=0, scale=0.375, size=clients_grad_dict[0][key].shape)).type(
                    torch.float).to(torch.device("cpu"))

                if c < 15:
                    noise = noise0

                t = clients_grad_dict[c][key] + noise
                temp.append(t)
                if c > 14:
                    client_average_grad_dict[key] += t / 50
                t_vector.append(np.concatenate((np.array(t.view(1, -1))), axis=0))
            # '''
            hv[epoch] = t_vector.copy()

            if key == 'fc3.weight':
                for l in range(len(t_vector)):
                    for n in range(epoch-1):
                        t_vector[l] += hv[n][l]
                similaritys = cosine_similarity(t_vector)
                print(similaritys)
                v = []
                for i in range(Num_client):
                    tmax = -999
                    for j in range(Num_client):
                        if i != j and similaritys[i][j] > tmax:
                            tmax = similaritys[i][j]
                    v.append(tmax)

                for i in range(Num_client):
                    for j in range(Num_client):
                        if v[j] > v[i]:
                            similaritys[i][j] *= v[i]/v[j]

                alpha = []
                for i in range(Num_client):
                    tmax = -999
                    for j in range(Num_client):
                        if i != j and similaritys[i][j] > tmax:
                            tmax = similaritys[i][j]
                    alpha.append(1-tmax)
                max_alpha = max(alpha)
                for i in range(Num_client):
                    alpha[i] = alpha[i] / max_alpha
                    if alpha[i] != 1:
                        alpha[i] = math.log(abs(alpha[i]/(1-alpha[i])))+0.5
                    if alpha[i] > 1:
                        alpha[i] = 1
                    if alpha[i] < 0:
                        alpha[i] = 0

            # '''
        # 加载梯度
        params_modules_server = net.named_parameters()
        for params_module in params_modules_server:
            (name, params) = params_module
            params.grad = client_average_grad_dict[name]  # 用字典中存储的子模型的梯度覆盖server中的参数梯度
        optimizer_server.step()

    sum_loss /= client_len * Num_client
    losses.append(sum_loss)
    print('loss: %.03f' % sum_loss)
    # 每训练100个batch打印一次平均loss
    # sum_loss += loss_c0.item()
    # if i % 100 == 99:
    #     print('[%d, %d] loss: %.03f'
    #           % (epoch + 1, i + 1, sum_loss / 100))
    #     sum_loss = 0.0

    # 每跑完一次epoch测试一下准确率
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        accs.append(100 * correct / total)
        print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))


plt.figure()
plt.plot(range(len(accs)), accs)
plt.ylabel('accuracy')
plt.savefig('.//log//epoch_{}_iid.png'.format(epochs))

plt.figure()
plt.plot(range(len(losses)), losses)
plt.ylabel('loss')
plt.savefig('.//log//epoch_{}_iid_loss.png'.format(epochs))