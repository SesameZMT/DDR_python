from __future__ import print_function, division
import torch
import torch.nn as nn 
from torch import nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms 
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils import data
from PIL import Image

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

# 定义超参数 
input_size = 256  #图像的总尺寸28*28
num_classes = 5  #标签的种类数
num_epochs = 10  #训练的总循环周期
batch_size = 64  #一个撮（批次）的大小，64张图片

data_transform =transforms.Compose([
    transforms.Resize(256),         # 把图片resize为256*256    
    #transforms.RandomResizedCrop(224), # 随机裁剪224*224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 标准化
    ])
# 训练集
train_dataset = datasets.ImageFolder(root='DDR/train', transform=data_transform)  # 标签为{'cats':0, 'dogs':1}


# 测试集
test_dataset = datasets.ImageFolder(root='DDR/test', transform=data_transform)  



# 构建batch数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True , num_workers=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True , num_workers=32)

#  卷积网络模块构建
# - 一般卷积层，relu层，池化层可以写成一个套餐
# - 注意卷积最后结果还是一个特征图，需要把图转换成向量才能做分类或者回归任务
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # 输入大小 (1, 28, 28)
            nn.Conv2d(                      # 卷积层
                in_channels=3,              # 灰度图，输入特征图个数
                out_channels=25,            # 要得到多少个特征图，16个卷积核
                kernel_size=5,              # 卷积核大小，此时为5*5
                stride=1,                   # 步长
                padding=2,                  # 也就是加几圈0，如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),                              # 输出的特征图为 (16, 28, 28)
            nn.ReLU(),                      # relu层，激活函数
            nn.MaxPool2d(kernel_size=2),    # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(25, 50, 5, 1, 2),     # 输出 (32, 14, 14)
            nn.ReLU(),                      # relu层
            nn.MaxPool2d(2),                # 输出 (32, 7, 7)
        )
        self.out = nn.Linear(50 * 64 * 64, 5)   # 全连接层得到的结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten操作，结果为：(batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


# 准确率作为评估标准
def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1] 
    rights = pred.eq(labels.data.view_as(pred)).sum() 
    return rights, len(labels) 

# 训练网络模型
# 实例化
net = CNN()
if use_gpu:
    net = net.to(device)
#损失函数
criterion = nn.CrossEntropyLoss()
if use_gpu:
    criterion = criterion.to(device) 
         
#优化器
optimizer = optim.Adam(net.parameters(), lr=0.001) #定义优化器，普通的随机梯度下降算法

#开始训练循环
for epoch in range(num_epochs):
    #当前epoch的结果保存下来
    train_rights = [] 

    for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
        data = data.to(device)
        target = target.to(device)
        net.train()
        output = net(data)
        loss = criterion(output, target) 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        right = accuracy(output, target) 
        train_rights.append(right) 

    
        if batch_idx: 
            
            net.eval() 
            val_rights = [] 
            
            for (data, target) in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = net(data) 
                right = accuracy(output, target) 
                val_rights.append(right)
                
            #准确率计算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                batch_idx / len(train_loader), 
                loss.data, 
                train_r[0] / train_r[1], 
                val_r[0] / val_r[1]))