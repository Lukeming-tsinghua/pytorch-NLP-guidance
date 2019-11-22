import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):   
    def __init__(self, args):
        '''
        Your code here.
        Define a text CNN structure.
        
        Note that args.kernel_sizes is a list,
        so you may need to use nn.ModuleList.
        '''
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        E = args.embed_dim
        C = args.class_num
        KS = args.kernel_sizes
        in_chan = 1
        out_chan = args.kernel_num
        dropout = args.dropout
        
        self.embed = nn.Embedding(V, E)
        self.convs = nn.ModuleList([nn.Conv2d(in_chan, out_chan, (K, E)) for K in KS])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(KS)*out_chan,C)
        
    def forward(self, x):
        '''
        Your code here.
        Give the forward pass of the model.
        
        With multiple kernel sizes, the input for 
        fully connected layer can be the concatenation
        of feature maps of different kernel sizes.
        '''
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
        x = torch.cat(x, dim = 1)
        cat = self.dropout(x)
        
        return self.linear(cat)
        