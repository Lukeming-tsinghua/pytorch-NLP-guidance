# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)    # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签

#load data
train_X = mnist.train.images                #训练集样本
validation_X = mnist.validation.images      #验证集样本
train_X = np.vstack((train_X, validation_X))
test_X = mnist.test.images                  #测试集样本
#labels
train_Y = mnist.train.labels                #训练集标签
validation_Y = mnist.validation.labels      #验证集标签
train_Y = np.vstack((train_Y, validation_Y))
test_Y = mnist.test.labels                  #测试集标签

print(train_X.shape,train_Y.shape)          #输出训练集样本和标签的大小
print(test_Y.shape,test_Y.shape) 
#查看数据，例如训练集中第一个样本的内容和标签
#print(train_X[0])       
#print(train_Y[0])

#可视化样本，下面是输出了训练集中前20个样本
fig, ax = plt.subplots(nrows=4,ncols=5,sharex='all',sharey='all')
ax = ax.flatten()
for i in range(20):
    img = test_X[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#%%
import torch
# Define Net
class Net(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        y_pred = self.linear3(h2_relu)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H1, H2, D_out = train_X.shape[0], train_X.shape[1], 300, 300, train_Y.shape[1]

train_X = torch.Tensor(train_X)
train_Y = torch.LongTensor(train_Y)

# Construct our model by instantiating the class defined above
model = Net(D_in, H1, H2, D_out)

# Construct our loss function and an Optimizer. 
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

# Train
BATCH_SIZE = 128
for epoch in range(500):
    for start in range(0, len(train_X), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = train_X[start:end]
        batchY = train_Y[start:end]

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY.max(1)[1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Find loss on training data
    loss = loss_fn(model(train_X), train_Y.max(1)[1]).item()
    print('Epoch:', epoch, 'Loss:', loss)

# test
test_X = torch.Tensor(test_X)
test_Y = torch.LongTensor(test_Y)

with torch.no_grad():
    pred = model(test_X)
pred = pred.max(1, keepdim=True)[1]
print(pred)

# compute accuracy
correct = pred.eq(test_Y.max(1)[1].view_as(pred)).sum().item()
print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct,
        len(test_Y),
        100. * correct /len(test_Y)))

