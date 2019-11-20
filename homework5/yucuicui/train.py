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
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    
    epoch_loss,epoch_acc=0.,0.
    model.train()
    total_len=0.
    best_dev_acc=0.
    for epoch in range(1,args.epochs+1):
        for batch in train_iter:
            feature,target=batch.text,batch.label
            with torch.no_grad():
                feature.t_(), target.sub_(1)
            preds=model(feature).squeeze()
            
            loss = F.cross_entropy(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            corrects = (torch.max(preds, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * corrects/batch.batch_size
            
            epoch_loss +=loss.item()*len(target)
            epoch_acc +=acc.item()*len(target)
            total_len +=len(target)
            
            train_loss=epoch_loss/total_len
            train_acc=epoch_acc/total_len
            
            dev_loss,dev_acc = eval(dev_iter, model, args)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(model.state_dict(), "classification.pth")

            print("Epoch",epoch,"Train Loss",train_loss,"Train_acc",train_acc)
            print("Epoch",epoch,"dev Loss",dev_loss,"dev_acc",dev_acc)


def eval(dev_iter, model, args):
    '''
    your code here.
    evaluation of the model.
    
    Hint: To save the best model and do earily stopping,
    you need to return the evaluation accuracy to train function.
    '''
    epoch_loss,epoch_acc=0.,0.
    model.eval()
    total_len=0.
    with torch.no_grad():
        for batch in dev_iter:
            feature,target=batch.text,batch.label
            #with torch.no_grad():
            feature.t_(), target.sub_(1)
            preds=model(feature).squeeze()
                
            loss = F.cross_entropy(preds, target)
            corrects = (torch.max(preds, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * corrects/batch.batch_size
                
            epoch_loss +=loss.item()*len(target)
            epoch_acc +=acc.item()*len(target)
            total_len +=len(target)
            
            dev_loss=epoch_loss/total_len
            dev_acc=epoch_acc/total_len
    model.train()
    return dev_loss,dev_acc
    



