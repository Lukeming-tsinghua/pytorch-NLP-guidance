#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[141]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# In[142]:


train_data = datasets.MNIST(root = './mnist_data/', train = True, 
                            transform = transforms.Compose(
                                [transforms.ToTensor(),
                                 ]),download = True)
test_data = datasets.MNIST(root = './mnist_data/', train = False, 
                          transform = transforms.Compose(
                                [transforms.ToTensor(),
                                 ]))


# In[140]:


traindata=[d[0].data.cpu().numpy() for d in train_data]
testdata=[d[0].data.cpu().numpy() for d in test_data]


# In[120]:


len(traindata[0][0][0])


# In[4]:


train_mean=np.mean(traindata)
train_std=np.std(traindata)
test_mean=np.mean(testdata)
test_std=np.std(testdata)


# In[5]:


print(train_mean)
print(train_std)
print(test_mean)
print(test_std)


# In[6]:


traindata[0]


# In[93]:


train_data1 = datasets.MNIST(root = './mnist_data/', train = True, 
                            transform = transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Normalize([train_mean],[train_std])]))
test_data1 = datasets.MNIST(root = './mnist_data/', train = False, 
                          transform = transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Normalize([test_mean],[test_std])]))


# In[94]:


traindata1=[d[0].data.cpu().numpy() for d in train_data1]


# In[132]:


traindata1


# In[127]:


batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset = train_data1, 
                                          batch_size = batch_size,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data1,
                                         batch_size = batch_size)


# In[97]:


x,y=zip(*train_loader)


# In[98]:


print(len(x[0]))
print(len(x[0][0]))
#len(x[0][0][0])


# In[145]:


class Net(nn.Module):
    def __init__(self, D_in, H_1, H_2, H_3, D_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(D_in, H_1)
        self.fc2 = nn.Linear(H_1, H_2)
        self.fc3=nn.Linear(H_2,H_3)
        self.fc4 = nn.Linear(H_3, D_out)
    
    def forward(self, x):
        h_1 = F.relu(self.fc1(x))
        h_2 = F.relu(self.fc2(h_1))
        h_3 = F.relu(self.fc3(h_2))
        y_prediction = self.fc4(h_3)
        return y_prediction


# In[130]:


def train(model, device, lossF, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = lossF(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 
                100. * batch_idx / len(train_loader), loss.item()
            ))


# In[137]:


def test(model, device, lossF, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += lossF(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# In[147]:



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net(784, 300, 60, 30, 10).to(device)
lr = 0.01
optimizer=optim.Adam(model.parameters(), lr = lr)
lossF= nn.CrossEntropyLoss()#loss Function 从nn库调出 不能直接放入自定义函数 而F库里则可以，但注意两个库同样的函数命名不一样


# In[148]:


epochs = 2
for epoch in range(1, epochs + 1):
    train(model, device, lossF, train_loader, optimizer, epoch)
    test(model, device, lossF, test_loader)


# In[ ]:




