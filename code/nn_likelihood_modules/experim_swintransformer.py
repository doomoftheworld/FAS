from common_imports import torch, tqdm, copy
from common_use_functions import path_join
from experim_ResNet import pred_eval
from constant import torch_ext
from math import ceil

def train_network_without_valid_cosine_annealing(net, epochs, train_dataloader, test_dataloader, update_freq, optim, train_criterion, eval_criterion, pth, net_name, lr_scheduler=False, save_last=False):
    # Move modules to the device
    # net.to(device)
    net.cuda()
    train_criterion.cuda()
    eval_criterion.cuda()
    # Determine if we apply the learning rate scheduler
    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=ceil(len(train_dataloader)/update_freq)*epochs)
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

        _, train_acc, train_loss = pred_eval(net, eval_criterion, train_dataloader, evaluate=True, set_name='train')
        _, test_acc, test_loss = pred_eval(net, eval_criterion, test_dataloader, evaluate=True, set_name='test')
                
        epoch_stats = {"epoch":epoch+1, "train_loss": train_loss, "test_loss":test_loss, "test_acc":test_acc, "train_acc":train_acc}
        train_history.append(epoch_stats)
        
        print('epoch {}/{} training loss: {}, train accuracy: {}, test loss: {}, test accuracy: {}'.format(epoch_stats['epoch'],
                                                                                                            epochs, epoch_stats['train_loss'],
                                                                                                            epoch_stats['train_acc'],
                                                                                                            epoch_stats['test_loss'],
                                                                                                            epoch_stats['test_acc'] ))
        if train_acc > best_accuracy:
            print('best accuracy improve from {} to {}'.format(best_accuracy, train_acc))
            best_accuracy= train_acc
            best_net = copy.deepcopy(net)

    # Save the best model to the pth path
    torch.save(best_net, path_join(pth, net_name+torch_ext))
    if save_last:
        torch.save(net, path_join(pth, 'last_'+net_name+torch_ext))
    return train_history