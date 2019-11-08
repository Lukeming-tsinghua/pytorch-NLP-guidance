import jieba
from torchtext import data
import re
from torchtext.vocab import Vectors
import numpy as np

# create a tokenizer function
def tokenizer(text): 
    """
    Q1:
       Please define the tokenizer function and make sure 'words = ' .
       're' and 'jieba' are good tools in this part.
    """
    #regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9\`\~\@#$\^&\*\(\)\=∥{}\'\'\[\.\<\>\/\?\~\!\@\#\\\&\*\%\s+]')
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')#非中文数字字母的数
    text = re.sub(regex,'',text)#去除
    
    ci_words=[word for word in jieba.cut(text)]
    stop_words=get_stop_words()
    wordss=[]
    for word in ci_words:
        if word in stop_words:
            word=''
        else:
            word=word
        wordss.append(word)
    text=''.join(str(x) for x in wordss)
    
    words = list(text)
    return words

# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt', 'rb')
    """
    Q2:
       Please make a stop-word-list .

    """
    stop_words = []
    for line in file_object.readlines():
        line = line.strip()
        stop_words.append(bytes.decode(line))
    return stop_words

def load_data(args):
    print('加载数据中...')
    stop_words = get_stop_words() # 加载停用词表
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    '''
    text = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=127, stop_words=stop_words)
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
    
#提取可用字向量
    '''
    embeddings_index = {} #定义空字典
    wordm=[]
    f = open("homework_3/data/维基百科词向量",encoding="UTF-8")#打开一个文件
    for line in f.readlines():
        values = line.split() #将词分割开
        word=values[0]
        wordm.append(word)
        coefs = np.asarray(values[1:], dtype='float32')#定义数组
        embeddings_index[word] = coefs    
    f.close()
    dict1={}
    #将字向量提取出来
    for i in train.text:
        for j in i:
            if j in wordm:
                dict1[j]=embeddings_index[j]
    for i in val.text:
        for j in i:
            if j in wordm:
                dict1[j]=embeddings_index[j]
    f=open("homework_3/data/字典.txt","w",encoding="utf-8")
    embedding_Q=str(dict1)
    f.write(embedding_Q)
    f.close
    '''
    if args.static:
        
        text.build_vocab(train, val, vectors=Vectors(name="data/result.txt"))
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
    return train_iter, val_iter

