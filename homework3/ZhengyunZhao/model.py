# _*_ coding: utf_8 _*_
"""
Created on Fri Nov  1 19:51:23 2019

@author: LVXINPENG
"""

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
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.direction = int(args.bidirectional) + 1
        
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, 
                                                            freeze = not args.fine_tune)
        self.LSTM = nn.LSTM(args.embedding_dim, self.hidden_size, self.layer_num, 
                            batch_first = True, bidirectional = args.bidirectional)
        self.Linear = nn.Linear(self.hidden_size * self.direction, args.label_num)
       
    def forward(self, x):
        """
        Q4:
            Please define the forword propagation 
            you can view the dataset(tsv) for obtaining the dimension of data
            and make sure that 'out = '
        """
        embed = self.embedding(x)
        output, (hn, cn) = self.LSTM(embed)
        out = self.Linear(output[:,-1,:])
        return out

