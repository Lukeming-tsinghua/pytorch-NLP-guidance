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
        
        embed_num = args.embed_num
        embed_dim = args.embed_dim
        class_num = args.class_num
        channel_num = args.kernel_num
        kernel_size = args.kernel_sizes
        self.embedding = nn.Embedding(embed_num, embed_dim)
        self.conv = nn.ModuleList([nn.Conv2d(1, channel_num, (K, embed_dim)) for K in kernel_size])
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(kernel_size)*channel_num, class_num)
        

    def forward(self, x):
        '''
        Your code here.
        Give the forward pass of the model.
        
        With multiple kernel sizes, the input for 
        fully connected layer can be the concatenation
        of feature maps of different kernel sizes.
        '''
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] 
        x = torch.cat(x, 1)

        x = self.dropout(x) 
        output = self.linear(x)  
        return output
