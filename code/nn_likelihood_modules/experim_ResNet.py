from pytorch_training_preparation import *

"""
ResNet training functions

Note: The following functions could also be considered as general training and evaluation funcitons for the classificatio task.
"""

"""
In all the defined following functions, "pth" means the path to the file or the folder which contains the file
"""

def train_network(net, epochs, train_dataloader, valid_dataloader, optim, train_criterion, eval_criterion, pth, net_name, lr_scheduler=False, save_last=False):
    # Move modules to the device
    # net.to(device)
    net.cuda()
    train_criterion.cuda()
    eval_criterion.cuda()
    # Determine if we apply the learning rate scheduler
    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[40, 55], gamma=0.1) # [10, 15] for satellite images
    # Evaluation based on the accuracy
    best_accuracy = 0
    best_net = None
    train_history = []
    for epoch in range(epochs):
        net.train()
        for batch in tqdm(train_dataloader, desc='Epoch '+str(epoch+1)):
            feature, target= batch[0].float(), batch[1].long()
            # feature = feature.to(device)
            # target = target.to(device)
            feature = feature.cuda()
            target = target.cuda()
            optim.zero_grad()
            pred= net(feature)
            loss= train_criterion(pred, target)
            loss.backward()
            optim.step()
        if lr_scheduler:
            scheduler.step()
        
        _, val_acc, val_loss = pred_eval(net, eval_criterion, valid_dataloader, evaluate=True, set_name='valid')
        _, train_acc, train_loss = pred_eval(net, eval_criterion, train_dataloader, evaluate=True, set_name='train')
                
        epoch_stats = {"epoch":epoch+1, "train_loss": train_loss, "val_loss":val_loss, "val_acc":val_acc, "train_acc":train_acc}
        train_history.append(epoch_stats)
        
        print('epoch {}/{} training loss: {}, train accuracy: {}, valid loss: {}, valid accuracy: {}'.format(epoch_stats['epoch'], epochs, epoch_stats['train_loss'], epoch_stats['train_acc'], epoch_stats['val_loss'], epoch_stats['val_acc'] ))
        if val_acc > best_accuracy:
            print('best accuracy improve from {} to {}'.format(best_accuracy, val_acc))
            best_accuracy= val_acc
            best_net = copy.deepcopy(net)

    # Save the best model to the pth path
    torch.save(best_net, path_join(pth, net_name+torch_ext))
    if save_last:
        torch.save(net, path_join(pth, 'last_'+net_name+torch_ext))
    return train_history

def train_network_based_on_loss(net, epochs, train_dataloader, valid_dataloader, optim, train_criterion, eval_criterion, pth, net_name, lr_scheduler=False, save_last=False):
    # Move modules to the device
    # net.to(device)
    net.cuda()
    train_criterion.cuda()
    eval_criterion.cuda()
    # Determine if we apply the learning rate scheduler
    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[40, 55], gamma=0.1) # [10, 15] for satellite images
    # Evaluation based on the loss
    best_loss = 0
    best_net = None
    train_history = []
    for epoch in range(epochs):
        net.train()
        for batch in tqdm(train_dataloader, desc='Epoch '+str(epoch+1)):
            feature, target= batch[0].float(), batch[1].long()
            # feature = feature.to(device)
            # target = target.to(device)
            feature = feature.cuda()
            target = target.cuda()
            optim.zero_grad()
            pred= net(feature)
            loss= train_criterion(pred, target)
            loss.backward()
            optim.step()
        if lr_scheduler:
            scheduler.step()
        
        _, val_acc, val_loss = pred_eval(net, eval_criterion, valid_dataloader, evaluate=True, set_name='valid')
        _, train_acc, train_loss = pred_eval(net, eval_criterion, train_dataloader, evaluate=True, set_name='train')
                
        epoch_stats = {"epoch":epoch+1, "train_loss": train_loss, "val_loss":val_loss, "val_acc":val_acc, "train_acc":train_acc}
        train_history.append(epoch_stats)
        
        print('epoch {}/{} training loss: {}, train accuracy: {}, valid loss: {}, valid accuracy: {}'.format(epoch_stats['epoch'], epochs, epoch_stats['train_loss'], epoch_stats['train_acc'], epoch_stats['val_loss'], epoch_stats['val_acc'] ))
        if epoch == 0:
           best_loss= val_loss
           best_net = copy.deepcopy(net) 
        else:
            if best_loss > val_loss:
                print('best loss improve from {} to {}'.format(best_loss, val_loss))
                best_loss= val_loss
                best_net = copy.deepcopy(net)

    # Save the best model to the pth path
    torch.save(best_net, path_join(pth, net_name+torch_ext))
    if save_last:
        torch.save(net, path_join(pth, 'last_'+net_name+torch_ext))
    return train_history


def pred_eval(net, criterion, dataloader, evaluate=False, set_name='train'):
    # Move modules to the device
    # net.to(device)
    net.eval()
    net.cuda()
    criterion.cuda()
    # Evaluation
    preds = []
    correct, total, loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluation on '+set_name+' set'):
            #Generate predictions for given batch in dataloader
            feature, target= batch[0].float(), batch[1].long()
            # feature = feature.to(device)
            # target = target.to(device)
            feature = feature.cuda()
            target = target.cuda()
            pred = net(feature)
            if evaluate:
                loss += criterion(pred, target)
            
            #Pick predicted class by max prob
            _, predicted = torch.max(pred.data, 1)
            preds.append(predicted)
            
            #Evaluate against target
            if evaluate:
                correct += (predicted==target).sum().item()
                total += target.size(0)
    if evaluate:
        return torch.cat(preds), float(correct/total), float(loss/total)
    else:
        return torch.cat(preds)
    
def accuracy_eval(net, dataloader, set_name='train'):
    # Move modules to the device
    # net.to(device)
    net.eval()
    net.cuda()
    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluation on '+set_name+' set'):
            #Generate predictions for given batch in dataloader
            feature, target= batch[0].float(), batch[1].long()
            # feature = feature.to(device)
            # target = target.to(device)
            feature = feature.cuda()
            target = target.cuda()
            pred = net(feature)
            #Pick predicted class by max prob
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted==target).sum().item()
            total += target.size(0)
    print('Evaluated accuracy on the set', set_name,':', str((correct/total)*100)+'%')