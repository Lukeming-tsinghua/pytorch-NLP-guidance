import pickle
import torch
import torch.nn as nn
import numpy as np
import model
import matplotlib.pyplot as plt


def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
        return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']


X_train, Y_train, X_test, Y_test = load()
H, W, K = 28, 28, 10
train_num = np.shape(X_train)[0]
test_num = np.shape(X_test)[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
Y_train = torch.from_numpy(Y_train).long().to(device)
Y_test = torch.from_numpy(Y_test).long().to(device)

batch_size = 32
learning_rate = 0.0001
weight_decay = 0.01
steps = 100
max_epoch = 3

def train(net, X_train, Y_train, train_num, device, max_epoch, batch_size, learning_rate, weight_decay):

    X_train = torch.reshape(X_train, (train_num, 1, H, W))
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    steps = int((train_num*0.8) // batch_size)
    net.train()
    for epoch in range(max_epoch):
        for step in range(steps):
            X_train_batch = X_train[step*batch_size:(step+1)*batch_size, ]
            Y_train_label_batch = Y_train[step*batch_size:(step+1)*batch_size, ]
            Y_train_pred_batch = net(X_train_batch).to(device)
            loss_batch = criterion(Y_train_pred_batch, Y_train_label_batch).to(device)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            if (step % 100 == 0):
                print([step, loss_batch])
    return net


def test(net, X_test, Y_test, test_num):

    X_test = torch.reshape(X_test, (test_num, 1, H, W))
    net.eval()
    with torch.no_grad():
        output = net(X_test,)
        Y_test_pred = torch.argmax(output, dim=1)
        count = np.sum(Y_test_pred.numpy() == Y_test.numpy())
        acc = count/test_num
        print(acc)


def main():

    net = model.Model()
    net.to(device)
    net = train(net, X_train, Y_train, train_num, device, max_epoch, batch_size, learning_rate, weight_decay)
    test(net, X_test, Y_test, test_num)
    # The final test accuracy is 94.25%


if __name__ == '__main__':
    main()
