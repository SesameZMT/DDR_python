# 经过调参结果：损失: 0.007205    训练集准确率: 100.00% 测试集正确率: 60.39%
from __future__ import print_function, division
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn , track
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision

from torch import nn
import torch.nn as nn 
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torchvision import datasets,transforms 

log_dir="tf-logs/1"
writer = SummaryWriter(log_dir=log_dir,flush_secs=10)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

# 定义超参数 
input_size = 256  #图像的总尺寸28*28
num_classes = 5  #标签的种类数
num_epochs = 30  #训练的总循环周期
batch_size = 300  #一个撮（批次）的大小，64张图片

data_transform =transforms.Compose([
    transforms.Resize(256),         # 把图片resize为256*256    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 标准化
    ])

# 训练集和测试集
train_dataset = datasets.ImageFolder(root='DDR/train', transform=data_transform)  
test_dataset = datasets.ImageFolder(root='DDR/test', transform=data_transform)  

# 构建batch数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True , num_workers=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True , num_workers=16)

class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, conv_1x1=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride) if conv_1x1 else None

    def forward(self, x):
        y = self.block(x)
        if self.conv_1x1:
            x = self.conv_1x1(x)
        return F.relu(y + x)


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block_2 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64),
            Residual(64, 128, stride=2, conv_1x1=True),
            Residual(128, 128),
            Residual(128, 256, stride=2, conv_1x1=True),
            Residual(256, 256),
            Residual(256, 512, stride=2, conv_1x1=True),
            Residual(512, 512),
            Residual(512, 1024, stride=2, conv_1x1=True),
            Residual(1024, 1024),
        )
        self.block_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(1024*16, 5),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x


def init_net(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1] 
    rights = pred.eq(labels.data.view_as(pred)).sum() 
    return rights, len(labels) 

net = ResNet()
if use_gpu:
    net = net.to(device)
net.apply(init_net)
#损失函数
criterion = nn.CrossEntropyLoss()
if use_gpu:
    criterion = criterion.to(device) 
#优化器
optimizer = optim.Adam(net.parameters(), lr=0.00005) #定义优化器，普通的随机梯度下降算法

di = 1
dt = 1
#开始训练循环
for epoch in range(num_epochs):
    name = "epoch_train"
    name1 = str(epoch)
    name = name + name1
    #当前epoch的结果保存下来
    with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TimeElapsedColumn()) as progress:
        batch_tqdm = progress.add_task(description=name, total=len(train_loader))
        train_rights = [] 
        r = 0
        k = 0
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
            r += right[0]
            k += right[1]
            writer.add_scalar('train_accuracy' , 100*r / k , di)
            writer.add_scalar('loss' , loss.data , di)
            di += 1
            progress.advance(batch_tqdm)    

    name2 = "epoch_test"
    name3 = str(epoch)
    name4 = name2 + name3
    with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TimeElapsedColumn()) as progress:
        test_tqdm = progress.add_task(description=name4, total=len(test_loader))
        rr = 0
        kk = 0
        net.eval() 
        val_rights = [] 
        for (data, target) in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data) 
            right = accuracy(output, target) 
            val_rights.append(right)
            rr = right[0]
            kk = right[1]
            writer.add_scalar('test_accuracy', 100*rr / kk , dt)
            dt += 1
            progress.advance(test_tqdm)
        #准确率计算
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

        print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
            epoch, batch_idx * batch_size, len(train_loader.dataset),
            100*batch_idx / len(train_loader), 
            loss.data, 
            100*train_r[0] / train_r[1], 
            100*val_r[0] / val_r[1]))
        
writer.close()