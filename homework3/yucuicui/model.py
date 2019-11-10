# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:51:23 2019

@author: LVXINPENG
"""

import torch
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, args):
        """
        Q3:
            Please define your model here.
            args is a bunch of parameters, 
            you can view them in train.py ,
            and use it just like ' self.hidden_size = args.hidden_size' .
        """
        
        super(TextRNN, self).__init__()#初始化模型
        embedding_dim = args.embedding_dim
        label_num = args.label_num
        vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.bidirectional = args.bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if args.static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.lstm = nn.LSTM(embedding_dim, 
                            self.hidden_size,
                            self.layer_num,
                            batch_first=True,
                            bidirectional=self.bidirectional) # 是否用双向
        self.fc = nn.Linear(self.hidden_size * 2, label_num) if self.bidirectional else nn.Linear(self.hidden_size, label_num)
       
    def forward(self, x):
        """
        Q4:
            Please define the forword propagation 
            you can view the dataset(tsv) for obtaining the dimension of data
            and make sure that 'out = '
        """
        input = self.embedding(x)
        #output,self.hidden=self.lstm(input,self.hidden)
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(self.layer_num, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(self.layer_num, x.size(0), self.hidden_size)
        output,(h_t,c_t) = self.lstm(input,(h0,c0))
        out = self.out(output[:,-1,:])#输出最后一步的结果
        return out

