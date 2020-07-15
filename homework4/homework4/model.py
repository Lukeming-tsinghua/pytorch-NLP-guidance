import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """ 一个简单的循环神经网络"""

    def __init__(self, vocab_size, embed_size, hidden_size):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(LSTM)
            - 一个线性层，从hidden state到输出单词表
            Write your code here
        '''

    
    

    def forward(self, text, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
            Write your code here
        '''

    
    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())
        return (weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad),
                weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad))
 

model = RNNModel(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)