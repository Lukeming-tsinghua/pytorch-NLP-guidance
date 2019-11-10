import jieba
from torchtext import data
import re
from torchtext.vocab import Vectors

# create a tokenizer function
def tokenizer(text): 
    """
    Q1:
       Please define the tokenizer function and make sure 'words = ' .
       're' and 'jieba' are good tools in this part.
    """
    words = jieba.lcut(text)
    return words

# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt', 'rb')
    """
    Q2:
       Please make a stop-word-list .

    """
    stop_words = []
    for line in file_object:
        stop_words.append(line)
    file_object.close()
    return stop_words

def load_data(args):
    print('加载数据中...')
    stop_words = get_stop_words() # 加载停用词表
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    '''
    text = data.Field(sequential=True, lower=True, tokenize=tokenizer, stop_words=stop_words, fix_length=127)
    label = data.Field(sequential=False)

    text.tokenize = tokenizer
    train, val = data.TabularDataset.splits(
            path='data/',
            skip_header=True,
            train='train.tsv',
            validation='validation.tsv',
            format='tsv',
            fields=[('index', None), ('label', label), ('text', text)],
        )
    #print(train[0])

    if args.static:
        
        text.build_vocab(train, val, vectors=Vectors(name="data/glove.6B.200d.txt"))
        args.embedding_dim = text.vocab.vectors.size()[-1]
        args.vectors = text.vocab.vectors
        
        """
        上面是我尝试使用的词向量(在data文件夹下），如果大家有更好的想法，可以在此处改为自己的词向量
        Optional question:
           Please give the arg.vectors here .
           Of course you can use the vocab.vectors above .
           You are more than welcome to give a better solution .
        """

    else: 
    	text.build_vocab(train, val)

    label.build_vocab(train, val)

    train_iter, val_iter = data.Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.text),
            batch_sizes=(args.batch_size, len(val)), # 训练集设置batch_size,验证集整个集合用于测试
            device=-1
    )
    args.vocab_size = len(text.vocab)
    args.label_num = len(label.vocab)
    #print(label.vocab.stoi)
    return train_iter, val_iter

'''
import argparse
parser = argparse.ArgumentParser(description='TextRNN text classifier')

parser.add_argument('-lr', type=float, default=0.01, help='学习率')
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=20)
parser.add_argument('-embedding-dim', type=int, default=128, help='词向量的维度')
parser.add_argument('-hidden_size', type=int, default=64, help='lstm中神经单元数')
parser.add_argument('-layer-num', type=int, default=1, help='lstm stack的层数')
parser.add_argument('-label-num', type=int, default=2, help='标签个数')
parser.add_argument('-bidirectional', type=bool, default=False, help='是否使用双向lstm')
parser.add_argument('-static', type=bool, default=False, help='是否使用预训练词向量')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=100, help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')

args = parser.parse_args()

train_iter, val_iter = load_data(args)
batch = next(iter(train_iter))
print(batch)
print(next(iter(train_iter)).text)
print(args.label_num)
'''