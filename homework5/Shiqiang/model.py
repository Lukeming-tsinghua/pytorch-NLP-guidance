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
        
        Vocab = args.embed_num # 已知词的数量
        Dim = args.embed_dim  #每个词向量长度
        Cla = args.class_num #类别数
        Ci = 1
        Knum = args.kernel_num 
        Ks = args.kernel_sizes  # 卷积核list，形如[2,3,4]

        self.embed = nn.Embedding(Vocab,Dim)
        self.convs = nn.ModuleList([nn.Conv2d(Ci,Knum,(K,Dim)) for K in Ks]) ## 卷积层
        self.dropout = nn.Dropout(args.dropout) 
        self.fc = nn.Linear(len(Ks)*Knum,Cla) ##全连接层

    def forward(self, x):
        '''
        Your code here.
        Give the forward pass of the model.
        
        With multiple kernel sizes, the input for 
        fully connected layer can be the concatenation
        of feature maps of different kernel sizes.
        '''
        #print(x.size())
        x = self.embed(x) #(N,W,D)

        x = x.unsqueeze(1) #(N,Ci,W,D)即(N,1,W,D)增加一维，变成类似图片的格式
        # 注意：卷积核的大小是（K，Dim），其中Dim就是词向量的长度，所以经过卷积后此维度会变为1
        # conv(x):（N，Knum，W‘，1） 其中，W‘大小是W-K+1 
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # len(Ks)*(N,Knum,W’)
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        
        x = torch.cat(x,1) #(N,Knum*len(Ks))  
        # A:(2,3) B:(4,3) C:(2,4) 
        # torch.cat((A,B),0)#按维数0（行）拼接 ->(6,3)
        # torch.cat((A,C),1)#按维数1（列）拼接 ->(2,7)
        
        x = self.dropout(x)
        logit = self.fc(x)
        return logit

