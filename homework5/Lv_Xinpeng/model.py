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
        super(CNN_Text,self).__init__()
        self.args = args
        
        embed_dim = args.embed_dim
        kernel_num = args.kernel_num
        kernel_sizes = args.kernel_sizes
        embed_num = args.embed_num
        class_num = args.class_num
        
        self.embed = nn.Embedding(embed_num, embed_dim) 
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])
        
        self.dropout = nn.Dropout(args.dropout)
        
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, class_num)
        

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
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        
        x = self.dropout(x)  
        out = self.fc1(x)  
        return out
