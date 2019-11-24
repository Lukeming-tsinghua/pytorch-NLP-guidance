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
        self.embed = nn.Embedding(args.embed_num,args.embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1,args.kernel_num,(k,args.embed_dim)) for k in args.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout) 
        self.linear = nn.Linear(len(args.kernel_sizes)*args.kernel_num, args.class_num) 


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
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x] 
        x = torch.cat(x,1) 
        x = self.dropout(x)
        logit = self.linear(x)
        #print(logit.shape)
        return logit