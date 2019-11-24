import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir,save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix,steps)
    torch.save(model.state_dict(),save_path)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            #print(type(feature.data))
            #print(feature.data.shape)
            f = torch.t(feature.data)        
            optimizer.zero_grad()
            logit = model(f)
            target.data = target.data-1
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                result = torch.max(logit,1)[1].view(target.size())
                corrects = (result.data == target.data).sum()
                accuracy = corrects*100.0/batch.batch_size
                sys.stdout.write('\rBatch[{}] - loss: {:.6f} acc: {:.4f}%({}/{})'.format(steps,
                                                                                        loss.data.item(),
                                                                                        accuracy,
                                                                                        corrects,
                                                                                        batch.batch_size))
            if steps % args.log_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model,args.save_dir,'best',steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model,args.save_dir,args.snapshot,steps)


def eval(dev_iter, model, args):
    '''
    your code here.
    evaluation of the model.
    
    Hint: To save the best model and do earily stopping,
    you need to return the evaluation accuracy to train function.
    '''
    model.eval()
    corrects, avg_loss = 0,0
    for batch in dev_iter:
        feature, target = batch.text, batch.label
        f = torch.t(feature.data)     
        logit = model(f)
        target.data = target.data-1
        loss = F.cross_entropy(logit,target)
        avg_loss += loss.data.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    size = len(dev_iter.dataset)
    avg_loss /= size 
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f} acc: {:.4f}%({}/{}) \n'.format(avg_loss,accuracy,corrects,size))
    
    return accuracy


