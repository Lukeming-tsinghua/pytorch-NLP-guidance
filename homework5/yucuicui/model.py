import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        '''
        Your code here.
        Define a text CNN structure.
        
        Note that args.kernel_sizes is a list,
        so you may need to use nn.ModuleList.
        '''
        super(CNN_Text,self).__init__()
        self.args=args
        embed_num=args.embed_num
        embed_dim=args.embed_dim
        class_num=args.class_num
        in_channels=1
        out_channels=args.kernel_num
        kernel_sizes=args.kernel_sizes
        
        self.embed=nn.Embedding(embed_num,embed_dim)
        self.convs=nn.ModuleList([nn.Conv2d(in_channels,out_channels,(K,embed_dim))for K in kernel_sizes])
        self.dropout=nn.Dropout(args.dropout)
        self.linear=nn.Linear(out_channels*len(kernel_sizes),class_num)
        
    def forward(self, x):
        '''
        Your code here.
        Give the forward pass of the model.
        
        With multiple kernel sizes, the input for 
        fully connected layer can be the concatenation
        of feature maps of different kernel sizes.
        '''
        x=self.embed(x)
#        x = x.permute(1,0,2) 
        x=x.unsqueeze(1)
        x=[F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x=[F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in x]
        x=torch.cat(x,dim=1)
        x=self.dropout(x)
        class_out=self.linear(x)
        return class_out