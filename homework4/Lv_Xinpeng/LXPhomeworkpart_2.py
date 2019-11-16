# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:43:36 2019

@author: lxpperfect
"""

import torchtext
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import numpy as np

#USE_CUDA = torch.cuda.is_available()
device = 'cuda'

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
MAX_VOCAB_SIZE = 50000
HIDDEN_SIZE=100

TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=".", 
    train="train.txt", validation="valid.txt", test="test.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)

VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=device, bptt_len=20, repeat=False, shuffle=True)


class RNNModel(nn.Module):
    """ 一个简单的循环神经网络"""

    def __init__(self, vocab_size, embed_size, hidden_size):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(LSTM)
            - 一个线性层，从hidden state到输出单词表
            Write your code here
        '''
        super(RNNModel,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    
    def forward(self, text, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
            Write your code here
        '''
        embed_vec = self.embedding(text)
        hidput, hidden = self.lstm(embed_vec, hidden)
        output = self.linear(hidput.view(-1, hidput.shape[2]))
        return output, hidden

    
    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())
        return (weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad, device=device),
                weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad, device=device))
 

model = RNNModel(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
model.to(device)


def repackage_hidden(h):
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

NUM_EPOCHS = 5
GRAD_CLIP = 5.0

def evaluate(model, data):
    model.eval()
    total_loss = 0.
    total_count = 0.
    it = iter(data)
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad = False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
            data.to(device)
            output, hidden = model(data, hidden)
            
            loss = loss_fn(output, target.view(-1))
            
            total_loss = loss.item() * np.multiply(*data.size())
            total_count = np.multiply(*data.size())
    
    loss = total_loss / total_count
    model.train()
    return loss

val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        hidden = repackage_hidden(hidden)
        data.to(device)
        output, hidden = model(data, hidden)
        
        loss  = loss_fn(output, target.view(-1))
        #print(type(loss))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
        optimizer.step()
        if i % 100 == 0:
            print("loss",loss.item())
            
        if i % 10000:
            val_loss = evaluate(model, val_iter)
            if len(val_losses) == 0 or val_loss< min(val_losses):
                torch.save(model.state_dict(),"lm.pth")
            else:
                # learning rate decay
                scheduler.step()
            val_losses.append(val_loss)
                
best_model = RNNModel(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
best_model.load_state_dict(torch.load("lm.pth"))

test_loss = evaluate(best_model, test_iter)
print("final score: ", np.exp(test_loss))