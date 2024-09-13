import copy
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from defined_data_structure import experiment_data
from common_imports import *

"""
Functions for the pratical neural network experimentation (training, evaluation)

All the following functions are the customized version for neural network training (mainly for the classification task performed 
with CNN and Inception), we programmed other simpler version of the training of specific architecture (ResNet. U-net, Autoencoder etc.)
And they both share a same module for the general training preparation (e.g. get optimizer, loss function) called "pytorch_training_preparation".

In all the following functions, the passed model should be moved to the indicated device in the function name

Note: This module is generally paired with the module "experim_preparation" (the customized functions for pytorch experiment preparation).
"""

"""
Version without the activation levels registration
"""
def train_gpu(model, train_loader, optimizer, loss_type='nll'):
    model.train()
    loss_fn = None
    if loss_type == 'nll':
        loss_fn = nn.NLLLoss(reduction='mean').cuda()
    elif loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(reduction='mean').cuda()
    else:
        print('please provide a correct loss function type.')
        exit(2)
    loop = tqdm_notebook(train_loader)
    for (data, target) in loop:
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output  = model(data.float())  # calls the forward function
        loss = loss_fn(output, target.long())
        loss.backward()
        optimizer.step()
    return model

def predict_gpu(model, data_loader):
    model.eval()
    final_pred = []
    with torch.no_grad():
          for data in data_loader:
                batch_data = Variable(data[0]).cuda()
                output = model(batch_data.float())
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                final_pred.extend(pred.cpu().reshape(-1).tolist())
    return final_pred

def predict_gpu_with_prob(model, data_loader):
    model.eval()
    final_pred = []
    with torch.no_grad():
          for data in data_loader:
                batch_data = Variable(data[0]).cuda()
                output = model(batch_data.float())
                final_pred.extend(output.data.cpu().tolist())
    return np.array(final_pred)

def evaluate_gpu(model, data_loader, set_name="test", loss_type='nll'):
    model.eval()
    eval_loss = 0
    correct = 0
    loss_fn = None
    if loss_type == 'nll':
        loss_fn = nn.NLLLoss(reduction='sum').cuda()
    elif loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(reduction='sum').cuda()
    else:
        print('please provide a correct loss function type.')
        exit(2)
    with torch.no_grad():
          for data, target in data_loader:
                data, target = Variable(data).cuda(), Variable(target).cuda()
                output = model(data.float())
                eval_loss += loss_fn(output, target.long()).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    size_dataset = len(data_loader.dataset)
    eval_loss = eval_loss/size_dataset
    accuracy = correct/size_dataset
    print('Loss of the', set_name, 'set :', eval_loss,', Accuracy on the ', set_name, 'set :', accuracy)
    return accuracy, eval_loss

def experiment_gpu(model, optim_type, lr, epochs, train_loader, valid_loader, test_loader, with_scheduler=False):
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = None
        change_optimizer = False
        if optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optim_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optim_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optim_type == 'sgd_momentum':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optim_type == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        elif optim_type == 'mix':
            """
            This mode trains the model with 4 epochs of Adam, then with SGD
            """
            optimizer = optim.Adam(model.parameters(), lr=lr)
            change_optimizer = True
        # Learning rate scheduler
        scheduler = None
        if with_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*epochs)], gamma=0.1)

        best_accuracy = 0
        best_model = None
        train_losses = []
        validation_losses = []
        train_accuracies = []
        validation_accuracies = []
        for epoch in range(epochs):
            if change_optimizer == True and epoch == 4:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                change_optimizer = False
            model = train_gpu(model, train_loader, optimizer)
            train_accu, train_loss = evaluate_gpu(model, train_loader, set_name='train')
            valid_accu, valid_loss = evaluate_gpu(model, valid_loader, set_name='validation')
            train_losses.append(train_loss)
            validation_losses.append(valid_loss)
            train_accuracies.append(train_accu)
            validation_accuracies.append(valid_accu)
            if valid_accu > best_accuracy:
                """
                 Use the validation set's accuracy to choose the best model
                """
                best_accuracy = valid_accu
                best_model = copy.deepcopy(model)
            # Learning rate scheduling
            if with_scheduler:
                scheduler.step()

        test_accu, test_loss = evaluate_gpu(best_model, test_loader, set_name='test')
        learning_data = experiment_data(lr,train_loader.batch_size,train_losses,validation_losses,train_accuracies,validation_accuracies,test_loss,test_accu,epochs) 

        return best_model, learning_data
    else:
        return None 
    
def experiment_gpu_without_valid(model, optim_type, lr, epochs, train_loader, test_loader, with_scheduler=False):
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = None
        change_optimizer = False
        if optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optim_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optim_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optim_type == 'sgd_momentum':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optim_type == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        elif optim_type == 'mix':
            """
            This mode trains the model with 4 epochs of Adam, then with SGD
            """
            optimizer = optim.Adam(model.parameters(), lr=lr)
            change_optimizer = True
        # Learning rate scheduler
        scheduler = None
        if with_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*epochs)], gamma=0.1)

        train_losses = []
        train_accuracies = []
        for epoch in range(epochs):
            if change_optimizer == True and epoch == 4:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                change_optimizer = False
            model = train_gpu(model, train_loader, optimizer)
            train_accu, train_loss = evaluate_gpu(model, train_loader, set_name='train')
            train_losses.append(train_loss)
            train_accuracies.append(train_accu)
            # Learning rate scheduling
            if with_scheduler:
                scheduler.step()

        test_accu, test_loss = evaluate_gpu(model, test_loader, set_name='test')
        learning_data = experiment_data(lr,train_loader.batch_size,train_losses,[],train_accuracies,[],test_loss,test_accu,epochs) 

        return model, learning_data
    else:
        return None

"""
Version with the activation levels registration (For the target registration, we register just the ground truth)
"""
def train_gpu_register_ver(model, train_loader, optimizer, loss_type='nll'):
    model.train()
    loss_fn = None
    if loss_type == 'nll':
        loss_fn = nn.NLLLoss(reduction='mean').cuda()
    elif loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(reduction='mean').cuda()
    else:
        print('please provide a correct loss function type.')
        exit(2)
    loop = tqdm_notebook(train_loader)
    nb_data = len(train_loader.dataset)
    registered_actLevel = {}
    registered_actLevel['class'] = []
    registered_actLevel['prob'] = []
    registered_actLevel['actLevel'] = {}
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = []
    for (data, target) in loop:
        # Registration of the targets
        registered_actLevel['class'].append(copy.deepcopy(target.numpy())) 
        
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output,actLevel  = model(data.float())  # calls the forward function
        loss = loss_fn(output, target.long())
        loss.backward()
        optimizer.step()

        # Registration of the activation levels
        for i, layer_actLevel in enumerate(actLevel):
            registered_actLevel['actLevel'][i].append(copy.deepcopy(layer_actLevel.detach().cpu().numpy())) 
        # Registration of the final probability
        registered_actLevel['prob'].append(copy.deepcopy(output.detach().cpu().numpy())) 
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = np.vstack(registered_actLevel['actLevel'][index])
        registered_actLevel['actLevel'][index] = registered_actLevel['actLevel'][index].reshape(nb_data,-1)
    registered_actLevel['prob'] = np.vstack(registered_actLevel['prob'])
    registered_actLevel['prob'] = registered_actLevel['prob'].reshape(nb_data,-1)
    registered_actLevel['class'] = np.hstack(registered_actLevel['class'])
    registered_actLevel['class'] = registered_actLevel['class'].reshape(nb_data,-1)
    return model, registered_actLevel

def evaluate_gpu_register_ver(model, data_loader, set_name="test", loss_type='nll'):
    model.eval()
    eval_loss = 0
    correct = 0
    loss_fn = None
    if loss_type == 'nll':
        loss_fn = nn.NLLLoss(reduction='sum').cuda()
    elif loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(reduction='sum').cuda()
    else:
        print('please provide a correct loss function type.')
        exit(2)
    nb_data = len(data_loader.dataset)
    registered_actLevel = {}
    registered_actLevel['class'] = []
    registered_actLevel['prob'] = []
    registered_actLevel['actLevel'] = {}
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = []
    with torch.no_grad():
          for data, target in tqdm(data_loader, desc='Processed batches'):
                # Registration of the targets
                registered_actLevel['class'].append(copy.deepcopy(target.numpy())) 
                
                data, target = Variable(data).cuda(), Variable(target).cuda()
                output,actLevel = model(data.float())
                eval_loss += loss_fn(output, target.long()).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                # Registration of the activation levels
                for i, layer_actLevel in enumerate(actLevel):
                    registered_actLevel['actLevel'][i].append(copy.deepcopy(layer_actLevel.detach().cpu().numpy())) 
                # Registration of the final probability
                registered_actLevel['prob'].append(copy.deepcopy(output.detach().cpu().numpy())) 
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = np.vstack(registered_actLevel['actLevel'][index])
        registered_actLevel['actLevel'][index] = registered_actLevel['actLevel'][index].reshape(nb_data,-1)
    registered_actLevel['prob'] = np.vstack(registered_actLevel['prob'])
    registered_actLevel['prob'] = registered_actLevel['prob'].reshape(nb_data,-1)
    registered_actLevel['class'] = np.hstack(registered_actLevel['class'])
    registered_actLevel['class'] = registered_actLevel['class'].reshape(nb_data,-1)
    size_dataset = len(data_loader.dataset)
    eval_loss = eval_loss/size_dataset
    accuracy = correct/size_dataset
    print('Loss of the', set_name, 'set :', eval_loss,', Accuracy on the ', set_name, 'set :', accuracy)
    return accuracy, eval_loss, registered_actLevel

def experiment_gpu_register_ver(model, optim_type, lr, epochs, train_loader, valid_loader, test_loader, with_scheduler=False):
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = None
        change_optimizer = False
        if optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optim_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optim_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optim_type == 'sgd_momentum':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optim_type == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        elif optim_type == 'mix':
            """
            This mode trains the model with 4 epochs of Adam, then with SGD
            """
            optimizer = optim.Adam(model.parameters(), lr=lr)
            change_optimizer = True
        # Learning rate scheduler
        scheduler = None
        if with_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*epochs)], gamma=0.1)

        best_accuracy = 0
        best_model = None
        train_losses = []
        validation_losses = []
        train_accuracies = []
        validation_accuracies = []
        actLevel_history = {}
        actLevel_history['training'] = {}
        for epoch in range(epochs):
            if change_optimizer == True and epoch == 4:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                change_optimizer = False
            model, registred_during_train_actLevel = train_gpu_register_ver(model, train_loader, optimizer)
            train_accu, train_loss, registred_train_actLevel = evaluate_gpu_register_ver(model, train_loader, set_name='train')
            valid_accu, valid_loss, registred_valid_actLevel = evaluate_gpu_register_ver(model, valid_loader, set_name='validation')
            train_losses.append(train_loss)
            validation_losses.append(valid_loss)
            train_accuracies.append(train_accu)
            validation_accuracies.append(valid_accu)
            # Save the current epoch's history
            current_epoch_history = {}
#             current_epoch_history['during_training'] = copy.deepcopy(registred_during_train_actLevel)
#             current_epoch_history['train'] = copy.deepcopy(registred_train_actLevel)
#             current_epoch_history['valid'] = copy.deepcopy(registred_valid_actLevel)
            current_epoch_history['during_training'] = registred_during_train_actLevel
            current_epoch_history['train'] = registred_train_actLevel
            current_epoch_history['valid'] = registred_valid_actLevel
            actLevel_history['training'][epoch] = current_epoch_history
            if valid_accu > best_accuracy:
                """
                 Use the validation set's accuracy to choose the best model
                """
                best_accuracy = valid_accu
                best_model = copy.deepcopy(model)
            # Learning rate scheduling
            if with_scheduler:
                scheduler.step()

        test_accu, test_loss, registred_test_actLevel = evaluate_gpu_register_ver(best_model, test_loader, set_name='test')
        learning_data = experiment_data(lr,train_loader.batch_size,train_losses,validation_losses,train_accuracies,validation_accuracies,test_loss,test_accu,epochs) 
        
        # Save the test history
        actLevel_history['test'] = copy.deepcopy(registred_test_actLevel)
        
        # Save the batch size
        actLevel_history['train_batch_size'] = train_loader.batch_size
        actLevel_history['valid_batch_size'] = valid_loader.batch_size
        actLevel_history['test_batch_size'] = test_loader.batch_size
        actLevel_history['model_name'] = model.model_name
        return best_model, learning_data, actLevel_history
    else:
        return None    

"""
Version with the activation levels registration (For the target registration, we register the ground truth and the predicted class)
"""
def evaluate_gpu_register_ver_with_predicts(model, data_loader, set_name="test", loss_type='nll'):
    model.eval()
    eval_loss = 0
    correct = 0
    loss_fn = None
    if loss_type == 'nll':
        loss_fn = nn.NLLLoss(reduction='sum').cuda()
    elif loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(reduction='sum').cuda()
    else:
        print('please provide a correct loss function type.')
        exit(2)
    nb_data = len(data_loader.dataset)
    registered_actLevel = {}
    registered_actLevel['class'] = []
    registered_actLevel['predict_class'] = []
    registered_actLevel['prob'] = []
    registered_actLevel['actLevel'] = {}
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = []
    with torch.no_grad():
          for data, target in tqdm(data_loader, desc='Processed batches'):
                # Registration of the targets (ground truth)
                registered_actLevel['class'].append(copy.deepcopy(target.numpy())) 
                
                data, target = Variable(data).cuda(), Variable(target).cuda()
                output,actLevel = model(data.float())
                eval_loss += loss_fn(output, target.long()).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                

                # Registration of the predictions
                registered_actLevel['predict_class'].append(copy.deepcopy(pred.detach().cpu().numpy().reshape(-1))) 
                # Registration of the activation levels
                for i, layer_actLevel in enumerate(actLevel):
                    registered_actLevel['actLevel'][i].append(copy.deepcopy(layer_actLevel.detach().cpu().numpy())) 
                # Registration of the final probability
                registered_actLevel['prob'].append(copy.deepcopy(output.detach().cpu().numpy())) 
    # Numpy array transformation
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = np.vstack(registered_actLevel['actLevel'][index])
        registered_actLevel['actLevel'][index] = registered_actLevel['actLevel'][index].reshape(nb_data,-1)
    registered_actLevel['prob'] = np.vstack(registered_actLevel['prob'])
    registered_actLevel['prob'] = registered_actLevel['prob'].reshape(nb_data,-1)
    registered_actLevel['class'] = np.hstack(registered_actLevel['class'])
    registered_actLevel['class'] = registered_actLevel['class'].reshape(nb_data,-1)
    registered_actLevel['predict_class'] = np.hstack(registered_actLevel['predict_class'])
    registered_actLevel['predict_class'] = registered_actLevel['predict_class'].reshape(nb_data,-1)
    size_dataset = len(data_loader.dataset)
    eval_loss = eval_loss/size_dataset
    accuracy = correct/size_dataset
    print('Loss of the', set_name, 'set :', eval_loss,', Accuracy on the ', set_name, 'set :', accuracy)
    return accuracy, eval_loss, registered_actLevel

"""
Models display
"""
def display_model(model):
    print(model)
    for name, parameter in model.named_parameters():
        print(name,":",parameter.size())
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters: ', model_total_params)


"""
Learning process display
"""
# Training process display function
def display(data, model_name, with_valid=True, just_graph=False):
    x_epochs = [x+1 for x in list(range(data.nb_epochs))]
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(x_epochs, data.train_losses, label="Train set")
    if with_valid:
        plt.plot(x_epochs, data.valid_losses, label="Validation set")
    plt.legend()
    plt.title(model_name+' (lr='+str(data.lr)+', batch_size='+str(data.batch_size)+')')
    plt.xlabel("Number of epochs")
    plt.ylabel("Average Negative log-likehood")

    plt.subplot(1,2,2)
    plt.plot(x_epochs, data.train_accuracies, label="Train set")
    if with_valid:
        plt.plot(x_epochs, data.valid_accuracies, label="Validation set")
    plt.legend()
    plt.title(model_name+' (lr='+str(data.lr)+', batch_size='+str(data.batch_size)+')')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.show()

    if just_graph == False:
        print('Loss of the test set :', data.test_loss,', Accuracy on the test set :', data.test_accuracy)

"""
CPU version
"""
def evaluate_cpu_register_ver(model, data_loader, set_name="test", loss_type='nll'):
    model.eval()
    eval_loss = 0
    correct = 0
    loss_fn = None
    if loss_type == 'nll':
        loss_fn = nn.NLLLoss(reduction='sum').cpu()
    elif loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(reduction='sum').cpu()
    else:
        print('please provide a correct loss function type.')
        exit(2)
    nb_data = len(data_loader.dataset)
    registered_actLevel = {}
    registered_actLevel['class'] = []
    registered_actLevel['prob'] = []
    registered_actLevel['actLevel'] = {}
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = []
    with torch.no_grad():
          for data, target in data_loader:
                # Registration of the targets
                registered_actLevel['class'].append(copy.deepcopy(target.numpy())) 
                
                data, target = Variable(data).cpu(), Variable(target).cpu()
                output,actLevel = model(data.float())
                eval_loss += loss_fn(output, target.long()).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                # Registration of the activation levels
                for i, layer_actLevel in enumerate(actLevel):
                    registered_actLevel['actLevel'][i].append(copy.deepcopy(layer_actLevel.detach().cpu().numpy())) 
                # Registration of the final probability
                registered_actLevel['prob'].append(copy.deepcopy(output.detach().cpu().numpy())) 
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = np.vstack(registered_actLevel['actLevel'][index])
        registered_actLevel['actLevel'][index] = registered_actLevel['actLevel'][index].reshape(nb_data,-1)
    registered_actLevel['prob'] = np.vstack(registered_actLevel['prob'])
    registered_actLevel['prob'] = registered_actLevel['prob'].reshape(nb_data,-1)
    registered_actLevel['class'] = np.hstack(registered_actLevel['class'])
    registered_actLevel['class'] = registered_actLevel['class'].reshape(nb_data,-1)
    size_dataset = len(data_loader.dataset)
    eval_loss = eval_loss/size_dataset
    accuracy = correct/size_dataset
    print('Loss of the', set_name, 'set :', eval_loss,', Accuracy on the ', set_name, 'set :', accuracy)
    return accuracy, eval_loss, registered_actLevel

def evaluate_cpu_register_ver_with_predicts(model, data_loader, set_name="test", loss_type='nll'):
    model.eval()
    eval_loss = 0
    correct = 0
    loss_fn = None
    if loss_type == 'nll':
        loss_fn = nn.NLLLoss(reduction='sum').cpu()
    elif loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(reduction='sum').cpu()
    else:
        print('please provide a correct loss function type.')
        exit(2)
    nb_data = len(data_loader.dataset)
    registered_actLevel = {}
    registered_actLevel['class'] = []
    registered_actLevel['predict_class'] = []
    registered_actLevel['prob'] = []
    registered_actLevel['actLevel'] = {}
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = []
    with torch.no_grad():
          for data, target in tqdm(data_loader, desc='Processed batches'):
                # Registration of the targets (ground truth)
                registered_actLevel['class'].append(copy.deepcopy(target.numpy())) 
                
                data, target = Variable(data).cpu(), Variable(target).cpu()
                output,actLevel = model(data.float())
                eval_loss += loss_fn(output, target.long()).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                

                # Registration of the predictions
                registered_actLevel['predict_class'].append(copy.deepcopy(pred.detach().cpu().numpy().reshape(-1))) 
                # Registration of the activation levels
                for i, layer_actLevel in enumerate(actLevel):
                    registered_actLevel['actLevel'][i].append(copy.deepcopy(layer_actLevel.detach().cpu().numpy())) 
                # Registration of the final probability
                registered_actLevel['prob'].append(copy.deepcopy(output.detach().cpu().numpy())) 
    # Numpy array transformation
    for index in range(model.nb_hidden_layers):
        registered_actLevel['actLevel'][index] = np.vstack(registered_actLevel['actLevel'][index])
        registered_actLevel['actLevel'][index] = registered_actLevel['actLevel'][index].reshape(nb_data,-1)
    registered_actLevel['prob'] = np.vstack(registered_actLevel['prob'])
    registered_actLevel['prob'] = registered_actLevel['prob'].reshape(nb_data,-1)
    registered_actLevel['class'] = np.hstack(registered_actLevel['class'])
    registered_actLevel['class'] = registered_actLevel['class'].reshape(nb_data,-1)
    registered_actLevel['predict_class'] = np.hstack(registered_actLevel['predict_class'])
    registered_actLevel['predict_class'] = registered_actLevel['predict_class'].reshape(nb_data,-1)
    size_dataset = len(data_loader.dataset)
    eval_loss = eval_loss/size_dataset
    accuracy = correct/size_dataset
    print('Loss of the', set_name, 'set :', eval_loss,', Accuracy on the ', set_name, 'set :', accuracy)
    return accuracy, eval_loss, registered_actLevel