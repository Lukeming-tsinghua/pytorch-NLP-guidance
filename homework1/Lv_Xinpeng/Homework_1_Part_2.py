# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:06:03 2019

@author: LVXINPENG
"""

#%%
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

data_path=u'D:/Code/mnistdata/'

def loaddata(filename):
    fp = open(filename, 'r')
    dataset = []
    labelset = []
    for i in fp.readlines():
        a = i.strip().split(',')
        #适当缩小数据
        dataset.append([(int(j) / 255.0 * 0.99 + 0.01) for j in a[1:]])  
        labelset.append(int(a[0]))
    dataset = torch.tensor(dataset)
    return dataset, np.array(labelset)


def transformlabel(labelset):
    new_labelset = []
    for i in labelset:
        new_label = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        new_label[i] = 0.99
        new_labelset.append(new_label)
    y = torch.tensor(new_labelset)
    return y

print("fuction is ok")

#%%
x_train, label_train = loaddata(data_path+'mnist_train.csv')
y_train = transformlabel(label_train)
print("trainset is ok")
x_test, label_test = loaddata(data_path+'mnist_test.csv')
y_test = transformlabel(label_test)
print("testset is ok")

#%%
D_in = len(x_train[0])
D_out = len(y_train[0]) 

model = nn.Sequential(
        nn.Linear(D_in, 100),
        nn.ReLU(),
        nn.Linear(100, D_out),
        #nn.ReLU(),
        #nn.Linear(100, 40),
        #nn.ReLU(),
        #nn.Linear(20, D_out)
        )

lr = 1e-3
loss_fn =  nn.MSELoss(reduction = "mean")
optimizer = optim.Adam(model.parameters(), lr = lr)

for i in range(521):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#%%
    
yt_pred = model(x_test)
rows = len(yt_pred)
rightcount = 0
for i in range(rows):
    for j in range(D_out):
        if round(yt_pred[i][j].item()) == round(y_test[i][j].item()):
            rightcount += 1
        else :
            rightcount = rightcount
print(rightcount)
print("正确率为:{}%".format(rightcount*100/ (rows*D_out)))

#得到的正确率为99.14%