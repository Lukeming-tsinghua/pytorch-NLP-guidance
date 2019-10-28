import torch 
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torch import  nn
from torch.autograd import Variable
from torch import  optim
from torchvision import transforms
 

train_set = mnist.MNIST('./data',train=True)
test_set = mnist.MNIST('./data',train=False)
 
# 预处理=>将各种预处理组合在一起
data_tf = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize([0.5],[0.5])])
 
train_set = mnist.MNIST('./data',train=True,transform=data_tf,download=True)
test_set = mnist.MNIST('./data',train=False,transform=data_tf,download=True)
 
train_data = DataLoader(train_set,batch_size=128,shuffle=True)
test_data = DataLoader(test_set,batch_size=128,shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=5), # 16, 24 ,24
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2,2))# 16, 12 ,12
        
        self.layer2 = nn.Sequential(
                nn.Conv2d(16,32,kernel_size=3),# 32, 10, 10
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2,2)) # 32,5,5
                
        self.fc = nn.Sequential(
                nn.Linear(32 * 5 * 5,512),
                nn.ReLU(),
                nn.Linear(512,10))
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
LR = 0.01
epoch = 5

net = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=LR,)
 

if __name__ == '__main__':
  for epoch in range(epoch):
    sum_loss = 0.0
    for i, data in enumerate(train_data):
      inputs, labels = data
      inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
      optimizer.zero_grad() 
      outputs = net(inputs) 
      loss = criterion(outputs, labels) 
      loss.backward() 
      optimizer.step() 
 
      # print(loss)
      sum_loss += loss.item()
      if i % 128 == 127:
        print('[%d,%d] loss:%.03f' %
           (epoch + 1, i + 1, sum_loss / 128))
        sum_loss = 0.0

net.eval() 
correct = 0
total = 0
for data_test in test_data:
    images, labels = data_test
    images, labels = Variable(images).cuda(), Variable(labels).cuda()
    output_test = net(images)
    _, predicted = torch.max(output_test, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("correct1: ", correct)
print("Test acc: {0}".format(correct.item() /
                 len(test_set)))