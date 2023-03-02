import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms 
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils import data
from PIL import Image


import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, datasets

input_size = 512  #图像的总尺寸28*28
num_classes = 5  #标签的种类数
num_epochs = 3  #训练的总循环周期
batch_size = 32  #一个撮（批次）的大小，64张图片


data_transform = transforms.Compose([
    transforms.Resize(512),         # 把图片resize为256*256    # 随机裁剪224*224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 标准化
])

train_dataset = datasets.ImageFolder(root='/Users/lixiaoyi/Desktop/大作业/Python大作业：深度学习解决图像视觉/教学/DDR的副本/train', transform=data_transform)  # 标签为{'cats':0, 'dogs':1}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = datasets.ImageFolder(root='/Users/lixiaoyi/Desktop/大作业/Python大作业：深度学习解决图像视觉/教学/DDR的副本/test', transform=data_transform)  
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(train_dataset[0])