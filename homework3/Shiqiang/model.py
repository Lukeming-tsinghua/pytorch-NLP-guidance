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
        self.embed_size = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.batch_size = args.batch_size
        self.num_direction = int(args.bidirectional) + 1
        
        self.embed = nn.Embedding(args.vocab_size, self.embed_size)
        if args.static:  
            self.embed = self.embed.from_pretrained(args.vectors, freeze=not args.fine_tune)
            
        self.lstm = nn.LSTM(self.embed_size,
                                self.hidden_size,
                                self.layer_num,
                                batch_first = True,
                                bidirectional = args.bidirectional)       
        self.out = nn.Linear(self.hidden_size * self.num_direction,args.label_num)
        

    def forward(self, x):
        """
        Q4:
            Please define the forword propagation 
            you can view the dataset(tsv) for obtaining the dimension of data
            and make sure that 'out = '
        """
        
        input = self.embed(x)
        #output,self.hidden=self.lstm(input,self.hidden)
        output,(h_t,c_t) = self.lstm(input)
        out = self.out(output[:,-1,:])
        return out
    '''
    # 初始化隐藏层
    def init_hidden(self,batch_size):
        #weight = next(self.parameters())
        return (torch.zeros((self.layer_num * self.num_direction, self.batch_size, self.hidden_size), requires_grad=True),
                     torch.zeros((self.layer_num * self.num_direction, self.batch_size, self.hidden_size), requires_grad=True))
    '''   