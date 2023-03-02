from turtle import backward
import matplotlib
from tomlkit import inline_table
import torch
import torchvision
import numpy as np
import random
from torch import tensor
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import warnings
import datetime
from sklearn import preprocessing

warnings.filterwarnings("ignore")

features = pd.read_csv("/Users/lixiaoyi/Desktop/大作业/Python大作业：深度学习解决图像视觉/教学/气温预测/temp_data.csv")


#得到年月日
days = features["day"]
months = features["month"]
years = features["year"]
#转换成datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]


# 准备画图
# 指定默认风格
plt.style.use('fivethirtyeight')
# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)
# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')
# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')
# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')
# 我的逗逼朋友
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')
plt.tight_layout(pad=2)


# 独热编码
#自动读入并判断，将字符转换成数字
features = pd.get_dummies(features)


# 标签
labels = np.array(features['actual'])
# 在特征中去掉标签
features= features.drop('actual', axis = 1)
# 名字单独保存一下，以备后患
feature_list = list(features.columns)
# 转换成合适的格式
features = np.array(features)


#将数值标准化
#标准化后收敛速度更快
#且loss值更小
input_features = preprocessing.StandardScaler().fit_transform(features)


#将x、y转换成需要的格式
x = torch.tensor(input_features, dtype = float)
y = torch.tensor(labels, dtype = float)


# 权重参数初始化
#其中，weights所构建的14*128的矩阵代表把输入数据的14个特征变换为128个隐藏特征
#偏置参数与结果挂钩
#应当与隐藏特征数量一致，也就是对隐藏特征都进行微调
#weights2所做的是将隐藏特征转化成一个值
weights = torch.randn((14, 128), dtype = float, requires_grad = True) 
biases = torch.randn(128, dtype = float, requires_grad = True) 
weights2 = torch.randn((128, 1), dtype = float, requires_grad = True) 
biases2 = torch.randn(1, dtype = float, requires_grad = True) 
learning_rate = 0.001 
losses = []


for i in range(50000):
    # 计算隐层
    hidden = x.mm(weights) + biases
    # 加入激活函数
    hidden = torch.relu(hidden)
    # 预测结果
    predictions = hidden.mm(weights2) + biases2
    # 通计算loss
    loss = torch.mean((predictions - y) ** 2) 
    losses.append(loss.data.numpy())
    # 打印loss值
    if i % 10000 == 0:
        print('loss:', loss)
    #返向传播计算
    loss.backward()
    #更新参数
    weights.data.add_(- learning_rate * weights.grad.data)  
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)
    # 每次迭代都得记得清空
    # 不清空grad将会累加
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

print(predictions.shape)
print(predictions)