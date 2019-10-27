#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
print("PyTorch Version: ",torch.__version__)


# In[2]:


torch.manual_seed(53113) 
batch_size=32

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data1',  train=True,download=True,#如果ture，从网上自动下载
                   transform=transforms.Compose([                       
                       transforms.ToTensor()])))                   
#transform 接受一个图像返回变换后的图像的函数，相当于图像先预处理下
#常用的操作如 ToTensor, RandomCrop，Normalize等. RandomCrop是随机剪裁

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data1', train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor()])))


# In[4]:


traindata=[d[0].data.cpu().numpy() for d in train_loader]
testdata=[d[0].data.cpu().numpy() for d in test_loader]


# In[5]:


train_mean=np.mean(traindata)
train_std=np.std(traindata)
test_mean=np.mean(testdata)
test_std=np.std(testdata)


# In[6]:


print(train_mean)
print(train_std)
print(test_mean)
print(test_std)


# In[7]:


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data1',  train=True,
                   transform=transforms.Compose([                       
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)) ])),#标准化
    batch_size=batch_size,shuffle=True)##torch.utils.data.DataLoader在训练模型时使用到此函数，用来把训练数据分成多个batch，
#此函数每次抛出一个batch数据，直至把所有的数据都抛出，也就是个数据迭代器。


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data1', train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize((0.1325,), (0.3105,))])),
    batch_size=batch_size,shuffle=True)#shuffle=True，打乱数据


# In[24]:


class Net(nn.Module):
    def __init__(self,D_in, H1,H2,H3, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1, bias=False)
        self.linear2 = torch.nn.Linear(H1, H2, bias=False)
        self.linear3 = torch.nn.Linear(H2, H3, bias=False)
        self.linear4 = torch.nn.Linear(H3, D_out, bias=False)

    def forward(self, x):  
        #print(x.shape)  #手写数字的输入维度，(N,1,28,28), N为batch_size
        x = x.view(-1, 28*28)
        x1 = F.relu(self.linear1(x)) 
        
        x2 = F.relu(self.linear2(x1))
        x3 = F.relu(self.linear3(x2))
        y_pred = F.softmax(self.linear4(x3))       
        return y_pred


# In[49]:


model = Net(784, 200,60,50,10)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[50]:


model


# In[51]:


type(train_loader)


# In[52]:


def train(model, train_loader, optimizer, epoch, log_interval=100):
    model.train() #进入训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        #data=data.view(data.shape[0],-1)
        optimizer.zero_grad() #梯度归零
        output = model(data)  #输出的维度[N,10] 这里的data是函数的forward参数x
        loss = loss_fn(output, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}".format(
                epoch, 
                batch_idx * len(data), #100*32
                len(train_loader.dataset), #60000
                100. * batch_idx / len(train_loader), #len(train_loader)=60000/32=1875
                loss.item()
            ))
            #print(len(train_loader))


# In[53]:


def test(model, test_loader):
    model.eval() #进入测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data=data.view(data.shape[0],-1)
            output = model(data) 
            test_loss += loss_fn(output, target).item()# 
            pred = output.argmax(dim=1, keepdim=True) # 
            
            #print(target.shape) #torch.Size([32])
            #print(pred.shape) #torch.Size([32, 1])
            correct += pred.eq(target.view_as(pred)).sum().item()
            #pred和target的维度不一样
            #pred.eq()相等返回1，不相等返回0，返回的tensor维度(32，1)。

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# In[54]:


epochs = 2
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)


# In[ ]:




