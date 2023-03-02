from pathlib import Path
import requests
from turtle import backward
import matplotlib
from tomlkit import inline_table
import torch
import torchvision
import numpy as np
import random
from torch import tensor
import pandas as pd
from matplotlib import pyplot
from torch import optim
import warnings
import datetime
from sklearn import preprocessing
import pickle
import gzip
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# 一般情况下，
# 如果模型有可学习的参数，
# 最好用nn.Module，其他情况nn.functional相对更简单一些

#打开数据集
with gzip.open("/Users/lixiaoyi/Desktop/大作业/Python大作业：深度学习解决图像视觉/教学/分类问题/mnist.pkl.gz", "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


#将矩阵全部转化成tensor的格式
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

loss_func = F.cross_entropy#计算交叉熵，见文档《机器学习中的交叉熵》

def model(xb):
    return xb.mm(weights) + bias

bs = 32#一次挑选bs个数据进行传入，也就是batch
xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype = torch.float,  requires_grad = True) 
bs = 32
bias = torch.zeros(10, requires_grad=True)

class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out  = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

#自动生成一个batch大小的训练集
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:'+str(step), '验证集损失：'+str(val_loss))

def get_model():
    #得到模型以及优化器
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)


def loss_batch(model, loss_func, xb, yb, opt=None):
    #opt是指一个优化器
    #用于求梯度，并更新参数
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(100, model, loss_func, opt, train_dl, valid_dl)