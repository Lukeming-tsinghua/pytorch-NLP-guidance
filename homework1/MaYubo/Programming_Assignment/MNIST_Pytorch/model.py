import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,30,3)
        self.conv3 = nn.Conv2d(30,10,3)
        self.conv4 = nn.Conv2d(10,1,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(20*20,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output