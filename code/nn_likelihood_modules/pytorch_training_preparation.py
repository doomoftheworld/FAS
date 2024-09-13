from common_imports import *
from common_use_functions import path_join
from constant import torch_ext

"""
In all the defined following functions, "pth" means the path to the file or the folder which contains the file
"""
def get_optimizer(model, optim_type, lr):
    """
    This function returns different optimizer based on the parameter value
    """
    optimizer = None
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optim_type == 'sgd_momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer

def get_criterion(criterion_type, mean_reduction=True):
    """""
    This function returns different criterion based on the parameter value.

    mean_reduction: boolean defines if we want the mean reduction (return sum reduction if not)

    Note: The returned criterions are mainly for classification task.
    """
    # Determine the reduction type
    reduction_type = None
    if mean_reduction:
        reduction_type = 'mean'
    else:
        reduction_type = 'sum'

    # Get the corresponding criterion
    criterion = None
    if criterion_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(reduction=reduction_type)
    elif criterion_type == 'nnl':
        criterion = nn.NLLLoss(reduction=reduction_type)
    elif criterion_type == 'mse':
        criterion = nn.MSELoss(reduction=reduction_type)
    
    return criterion

def load_model_by_net_name(pth, net_name):
    # Load the wanted model
    return torch.load(path_join(pth, net_name+torch_ext))


def load_state_dict_by_state_dict_name(model, pth, state_dict_name):
    # Load the state dict to the desired model
    model.load_state_dict(torch.load(path_join(pth, state_dict_name+torch_ext)))
    return model

def create_loader_from_torch_dataset(dataset, batch_size=64, shuffle=True, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)