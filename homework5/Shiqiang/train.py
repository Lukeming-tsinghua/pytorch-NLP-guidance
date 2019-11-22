import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    '''
    your code here.
    
    training process using backpropagation.
    print training loss and accuracy at args.log_interval.
    print evaluation loss and accuray at args.test_interval.
    Save the best model.
    
    Hint: view the size of data from train_iter before using them.
    Optional: Implement early stopping/dropout/L2 penalty.
    '''
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    best_acc = 0.
    step = 0
    model.train()
    for epoch in range(args.epochs):
        #print("Epoch {}/{}".format(epoch, args.epochs-1))
        #print("-"*10)
        
        for batch in train_iter:
            feature, targets = batch.text, batch.label # (W,D)
            #  print(feature.size(),targets.size())  torch.Size([45, 64]) torch.Size([64])
            # 输入x的size：torch.Size([45, 64, 128])
            # 此时feature的每一列是一个句子，共64个句子
            with torch.no_grad():
                feature.data.t_(), targets.data.sub_(1)  # 转置,batch_size在中间,分类标签为1,2，需减去1
            #print(targets)
            output = model(feature)
            loss = F.cross_entropy(output, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % args.log_interval == 0:
                #loss = loss.item() * feature.size(0)
                
                corrects = (torch.max(output, 1)[1].view(targets.size()).data == targets.data).sum()
                acc = float(corrects)/float(batch.batch_size)*100.0
                print('\rTrain: Batch[{}] - loss: {:.6f} acc: {:.2f}%)'.format(step, loss.item(), acc))
            
            if step % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    torch.save(model.state_dict(), 'text-cnn.pt')


def eval(dev_iter, model, args):
    '''
    your code here.
    evaluation of the model.
    
    Hint: To save the best model and do earily stopping,
    you need to return the evaluation accuracy to train function.
    '''
    model.eval()
    
    dev_loss = 0.
    corrects = 0.

    for batch in dev_iter:
        feature, targets = batch.text, batch.label # (W,D)
        feature.data.t_(), targets.data.sub_(1)  # 转置
        output = model(feature)
        loss = F.cross_entropy(output, targets)

        dev_loss += loss.item() * feature.size(0)
        #preds = torch.max(output, 1)[1]
        #corrects += torch.sum(preds.view(-1) == targets.view(-1)).item()
        corrects = (torch.max(output, 1)[1].view(targets.size()).data == targets.data).sum()

    dev_loss = dev_loss/len(dev_iter.dataset)
    acc = float(corrects)/float(len(dev_iter.dataset))*100.0
    print('\rDev: loss: {:.6f} acc: {:.2f}%)'.format(dev_loss, acc))
    model.train()        
    return acc

