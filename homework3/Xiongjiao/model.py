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
        embedding_dim = args.embedding_dim
        label_num = args.label_num
        vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.bidirectional = args.bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if args.static:  
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.lstm = nn.LSTM(embedding_dim, 
                            self.hidden_size,
                            self.layer_num,
                            batch_first=True,
                            bidirectional=self.bidirectional) 
        self.Linear = nn.Linear(self.hidden_size * 2, label_num) if self.bidirectional else nn.Linear(self.hidden_size, label_num) 
        
       
    def forward(self, data):
        """
        Q4:
            Please define the forword propagation 
            you can view the dataset(tsv) for obtaining the dimension of data
            and make sure that 'out = '
        """
        data = self.embedding(data) 
        h0 = torch.zeros(self.layer_num * 2, data.size(0), self.hidden_size) if self.bidirectional else torch.zeros(self.layer_num, data.size(0), self.hidden_size)
        c0 = torch.zeros(self.layer_num * 2, data.size(0), self.hidden_size) if self.bidirectional else torch.zeros(self.layer_num, data.size(0), self.hidden_size)
        out, (hn, cn) = self.lstm(data, (h0, c0))
        out = self.Linear(out[:, -1, :])
        return out

