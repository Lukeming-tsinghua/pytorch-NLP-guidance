#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os


# In[2]:


batch_size = 64
learning_rate = 0.02
num_epoches = 10


# In[3]:


train_set = datasets.MNIST('./mnist/', train=True, transform = transforms.ToTensor())
test_set = datasets.MNIST('./mnist/', train=False, transform = transforms.ToTensor())


# In[4]:


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# In[5]:


class Net(nn.Module):
    def __init__(self, input_dim, num_hidden1, num_hidden2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, num_hidden1), nn.BatchNorm1d(num_hidden1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(num_hidden1, num_hidden2), nn.BatchNorm1d(num_hidden2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(num_hidden2, out_dim))
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 


# In[6]:


model = Net(784, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# In[7]:


# model training 
for epoch in range(num_epoches):
    batch = 0
    for img,label in train_loader:
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # forward 
        pred = model(img)
        loss = criterion(pred, label)
        print_loss = loss.data.item()
        
        optimizer.zero_grad()
        
        # backward
        loss.backward()
        optimizer.step()
        
        batch += 1
        if batch % 50 == 0:
            print('Epoch: {}, Batch: {}, loss: {:.4}'.format(epoch, batch, loss.data.item()))


# In[9]:


# model testing 
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
 
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_set)), eval_acc / (len(test_set))))


# In[ ]:




