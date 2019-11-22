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
    print evaluation loss and accuracy at args.test_interval.
    Save the best model.
    
    Hint: view the size of data from train_iter before using them.
    Optional: Implement early stopping/dropout/L2 penalty.
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    model.train()
    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            text, target = batch.text, batch.label
            with torch.no_grad():
                text.t_(), target.sub_(1)
            optimizer.zero_grad()
            output = model(text)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            steps += 1
            if steps % (args.log_interval*50) == 0:
                corrects = (torch.max(output, 1)[1].view(target.size()).data == 
                            target.data).sum()
                accuracy = 100.0 * float(corrects)/batch.batch_size
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})\n'
                                 .format(steps, loss.item(), accuracy, corrects, batch.batch_size))
#                print("Batch:", steps, "loss:", loss.item(), "acc:", accuracy)
                
            
            if steps % args.test_interval == 0:
                eval_loss, eval_acc = eval(dev_iter, model, args)
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    last_step = steps
                else:
                    if steps - last_step >= args.early_stop:
                        print("early stop by {} steps.".format(args.early_stop))
                
    

def eval(dev_iter, model, args):
    '''
    your code here.
    evaluation of the model.
    
    Hint: To save the best model and do earily stopping,
    you need to return the evaluation accuracy to train function.
    '''
    model.eval()
    corrects = 0. 
    avg_loss = 0.
    with torch.no_grad():
        for batch in dev_iter:
            text, target = batch.text, batch.label
            text.t_(), target.sub_(1)
            output = model(text)
            loss = F.cross_entropy(output, target)
            
            avg_loss += loss.item()
            corrects += (torch.max(output, 1)[1].view(target.size()).data 
                         == target.data).sum()
            
        size = len(dev_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * float(corrects)/size
        sys.stdout.write('\nEvaluation - loss: {:.6f} acc:{:.4f}%({}/{}) \n'.format
                         (avg_loss, accuracy, corrects, size))
#        print("Average loss:{}", avg_loss, "accuracy:", accuracy, "corrects:", corrects)
        
        return avg_loss, accuracy
