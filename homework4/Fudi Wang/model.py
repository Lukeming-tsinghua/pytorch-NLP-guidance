import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """ 一个简单的循环神经网络"""
    def __init__(self, vocab_size, embed_size, hidden_size, dropout = 0.5):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(LSTM)
            - 一个线性层，从hidden state到输出单词表
            Write your code here
        '''
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
    
    
    def forward(self, text, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
            Write your code here
        '''
        emb = self.drop(self.embed(text)) # seq_length * batch_size * embed_size
        output, hidden = self.lstm(emb, hidden) 
        output = self.drop(output)
        decoded = self.linear(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
    
    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())
        return (weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad),
                weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad))
 

model = RNNModel(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)