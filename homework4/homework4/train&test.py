def repackage_hidden(h):
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

NUM_EPOCHS = 3
GRAD_CLIP = 5.0

def evaluate(model, data):
    model.eval()
    total_loss = 0.
    total_count = 0.
    it = iter(data)
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad = False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

        '''
        这里写出loss的表达式，即需要向loss_fn传递的参数
        Write your code here
        '''
            total_loss = loss.item * np.multiply(*data.size())
            total_count = np.multiply(*data.size())
    
    loss = total_loss / total_count
    model.train()
    return loss

val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        
        '''
        这里写出loss的表达式，即需要向loss_fn传递的参数
        Write your code here
        '''
        print(type(loss))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
        optimizer.step()
        if i % 100 == 0:
            print("loss",loss.item())
            
        if i % 10000:
            val_loss = evaluate(model, val_iter)
            if len(val_losses) == 0 or val_loss< min(val_losses):
                torch.save(model.state_dict(),"lm.pth")
            else:
                # learning rate decay
                scheduler.step()
            val_losses.append(val_loss)
                
best_model = RNNModel(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
best_model.load_state_dict(torch.load("lm.pth"))

test_loss = evaluate(best_model, test_iter)
print("final score: ", np.exp(test_loss))