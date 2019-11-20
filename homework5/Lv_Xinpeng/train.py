import os
import sys
import torch
#import torch.autograd as autograd
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
    if args.cuda:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            with torch.no_grad():
                feature.t_()
                target.sub_(1)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            y_pred = model(feature)
            loss = F.cross_entropy(y_pred, target)
            loss.backward()
            optimizer.step()
            
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(y_pred, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/float(batch.batch_size) * 100.0
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.4f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                eval(dev_iter, model, args)
                
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)


def eval(dev_iter, model, args):
    '''
    your code here.
    evaluation of the model.
    
    Hint: To save the best model and do earily stopping,
    you need to return the evaluation accuracy to train function.
    '''
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in dev_iter:
        feature, target = batch.text, batch.label
        with torch.no_grad():
            feature.t_()
            target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        y_pred = model(feature)
        loss = F.cross_entropy(y_pred, target, size_average=False)
        avg_loss += loss.item()
        corrects += (torch.max(y_pred,1)[1].view(target.size()).data==target.data).sum()
        
        size = len(dev_iter.dataset)
        avg_loss = loss.item()/size
        accuracy = float(corrects)/float(size) * 100.0
        model.train()
        print('\nEvaluation - loss: {:.4f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))



