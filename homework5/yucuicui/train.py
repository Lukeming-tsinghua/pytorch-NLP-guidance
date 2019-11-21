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
    
    #epoch_loss,epoch_acc=0.,0.
    model.train()
    total_len=0.
    best_dev_acc=0.
    steps=0
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
            steps +=1   
            if steps % args.test_interval == 0:
                corrects = (torch.max(preds, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/float(batch.batch_size) * 100.0
                sys.stdout.write('\rBatch[{}] - loss: {:.4f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
                dev_loss,dev_acc,corrects,total_len = eval(dev_iter, model, args)
                sys.stdout.write('\nEvaluation - loss: {:.4f}  acc: {:.4f}%({}/{}) \n'.format(dev_loss, 
                                                                       dev_acc, 
                                                                       corrects, 
                                                                       total_len))
                print(" "*10)
            '''           
            epoch_loss +=loss.item()
            epoch_acc +=acc.item()*len(target)
            total_len +=len(target)
            
            train_loss=epoch_loss/total_len
            train_acc=epoch_acc/total_len
            '''           
            if steps % args.test_interval == 0:
                dev_loss,dev_acc,corrects,total_len = eval(dev_iter, model, args)
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    torch.save(model.state_dict(), "classification.pth")
                
            
            '''            
            if steps % 10 ==0:
                print("Epoch",epoch,"step",steps,"Train Loss",train_loss,"Train_acc",train_acc)
                print("Epoch",epoch,"step",steps,"dev Loss",dev_loss,"dev_acc",dev_acc)
                print(" "*10)

            '''
def eval(dev_iter, model, args):
    '''
    your code here.
    evaluation of the model.
    
    Hint: To save the best model and do earily stopping,
    you need to return the evaluation accuracy to train function.
    '''
    epoch_loss,epoch_acc=0.,0.
    total_len=0.
    model.eval()
    with torch.no_grad():
        for batch in dev_iter:
            feature,target=batch.text,batch.label
            #with torch.no_grad():
            feature.t_(), target.sub_(1)
            preds=model(feature).squeeze()
                
            loss = F.cross_entropy(preds, target,size_average=False)
            
            corrects = (torch.max(preds, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * corrects
              
            epoch_loss +=loss.item()
            epoch_acc +=acc.item()
        total_len=len(dev_iter.dataset)
        dev_loss=epoch_loss/total_len
        dev_acc=epoch_acc/total_len

    model.train()
    return dev_loss,dev_acc,corrects,total_len
    



