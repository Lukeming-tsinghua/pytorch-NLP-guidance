#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn


# In[2]:


train_data = datasets.MNIST(root = './data/', train = True, 
                            transform = transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Normalize([0.131],[0.308])]),
								 download = True)
test_data = datasets.MNIST(root = './data/', train = False, 
                          transform = transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Normalize([0.133],[0.309])]))


# In[3]:


batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset = train_data, 
                                          batch_size = batch_size,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                         batch_size = batch_size)


# In[4]:


class Net(nn.Module):
    def __init__(self, D_in, H_1, H_2, D_out):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(D_in, H_1)
        self.Linear2 = nn.Linear(H_1, H_2)
        self.Linear3 = nn.Linear(H_2, D_out)
    
    def forward(self, x):
        h_1 = F.relu(self.Linear1(x))
        h_2 = F.relu(self.Linear2(h_1))
        y_pred = self.Linear3(h_2)
        return y_pred


# In[5]:


model = Net(784, 196, 49, 10)
if torch.cuda.is_available() == True:
    model = model.cuda()
crit = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)


# In[6]:


def train(epoch):
    for batch_id, (data, y) in enumerate(train_loader):
        data = data.view(data.shape[0], -1)
        if torch.cuda.is_available() == True:
            data = data.cuda()
            y = y.cuda()
        y_pred = model(data)
        loss = crit(y_pred, y)
        if batch_id % 100 == 0:
            print('Epoch: {}, Batch: {}, Loss:{}'.format(epoch, batch_id, loss.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# In[7]:


def test():
    accuracy = 0
    for idx, (data, y) in enumerate(test_loader):
        if torch.cuda.is_available() == True:
            data = data.cuda()
            y = y.cuda()
        prob = model(data.view(data.shape[0], -1))
        y_pred = prob.max(1, keepdim = True)[1]
        accuracy += y_pred.eq(y.view_as(y_pred)).sum()
    print (accuracy.item() / len(test_loader.dataset))


# In[8]:


NUM_EPOCH = 10
for i in range(NUM_EPOCH):
    train(i)


# In[9]:


test()


# In[ ]:




