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
        self.embedding_size = args.embedding_dim 
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.label_num = args.label_num
        self.bidirectional = args.bidirectional
        self.vocab_size = args.vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        if args.static:  
            self.embedding = self.embedding.from_pretrained(args.vectors,
                                                            freeze=not args.fine_tune)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.layer_num, 
                            batch_first = True, bidirectional = self.bidirectional)
        if self.bidirectional:
            self.linear = nn.Linear(self.hidden_size * 2, self.label_num)
        else:
            self.linear = nn.Linear(self.hidden_size, self.label_num)
        
       
    def forward(self, x):
        """
        Q4:
            Please define the forword propagation 
            you can view the dataset(tsv) for obtaining the dimension of data
            and make sure that 'out = '
        """
        embedding = self.embedding(x)
        if self.bidirectional:
            h0 = torch.randn(self.layer_num * 2, embedding.size(0), self.hidden_size)/20
            c0 = torch.randn(self.layer_num * 2, embedding.size(0), self.hidden_size)/20
        else:
            h0 = torch.randn(self.layer_num, embedding.size(0), self.hidden_size)/20
            c0 = torch.randn(self.layer_num, embedding.size(0), self.hidden_size)/20
        out, (hidden, cell) = self.lstm(embedding, (h0, c0))
        out = self.linear(out[:,-1,:])
        return out

