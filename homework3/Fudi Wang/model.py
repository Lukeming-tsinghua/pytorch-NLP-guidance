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
        super(TextRNN, self).__init__()
        self.vocab_size = args.vocab_size
        self.embed_size = args.embedding_dim
        self.label_num = args.label_num
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.bidirectional = args.bidirectional
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.layer_num, batch_first = True,
                            bidirectional = self.bidirectional)
        if self.bidirectional:
            self.linear = nn.Linear(self.hidden_size*2, self.label_num)
        else:
            self.linear = nn.Linear(self.hidden_size, self.label_num)
        
        
       
    def forward(self, x):
        """
        Q4:
            Please define the forword propagation 
            you can view the dataset(tsv) for obtaining the dimension of data
            and make sure that 'out = '  
        """
        embed = self.embed(x)
        
        # initialize h0 and c0
        if self.bidirectional:
            h0 = torch.zeros(self.layer_num * 2, embed.size(0), self.hidden_size)
            c0 = torch.zeros(self.layer_num * 2, embed.size(0), self.hidden_size)
        else:
            h0 = torch.zeros(self.layer_num, embed.size(0), self.hidden_size)
            c0 = torch.zeros(self.layer_num, embed.size(0), self.hidden_size)
        out, (hidden,  cell) = self.lstm(embed, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

