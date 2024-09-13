from common_imports import *
from common_use_functions import contents_of_folder, path_join, load_json, join_string, str_first_part_split_from_r
from constant import np_file_extension, model_config_file_keyword, model_state_file_keyword

"""
All the following functions are the customized version for network training preparation (mainly for the classification task performed 
with CNN and Inception), we programmed other simpler version for the general training preparation 
(e.g. get optimizer, loss function) called "pytorch_training_preparation".

Note: This module is generally paired with the module "experim_neural_network" (the customized functions for pytorch training experiment).
"""

"""
Function for the dataset loading
"""
def load_dataset(dataset_path, valid_set_name, display=True):
    """
    This function load all the .npy files exsited in the folder and verify the existence of a validation set
    """
    # Existence of the validation set
    is_valid_set = False

    # Load data sets
    files = contents_of_folder(dataset_path)
    loaded_datasets_dict = {}
    for file in files:
        if np_file_extension in file:
            current_file_name = file.split('.')[0]
            loaded_datasets_dict[current_file_name] = np.load(path_join(dataset_path, file))
        if valid_set_name in file:
            is_valid_set = True
    if display:
        print(loaded_datasets_dict)
        for key in loaded_datasets_dict:
            print(key,':',loaded_datasets_dict[key].shape)
        print('Existence of the validation set :', str(is_valid_set))
    return loaded_datasets_dict, is_valid_set

"""
Pytorch Data Loader Generation
"""
def create_dataloader(X, y, batch_size, shuffle=False, type_conversion=True):
    """
    This is a simple version to create dataloaders, the type conversion is only
    valid for classification tasks.
    """
    # The provided X and y should be numpy arrays
    # Inputs and labels
    torch_inputs = None

    torch_labels = None
    if type_conversion:
        torch_inputs = torch.from_numpy(X).float()
        torch_labels = torch.from_numpy(y).long()
    else:
        torch_inputs = torch.from_numpy(X)
        torch_labels = torch.from_numpy(y)
    # TensorDataset
    torch_dataset = TensorDataset(torch_inputs, torch_labels)
    # Generate the dataloader
    torch_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle) 
    return torch_loader

def create_dataloader_different_tasks(X, y, batch_size, shuffle=False, type_conversion='classification'):  
    """  
    Create a dataloader to from numpy arrays, this function could serve more tasks.

    X: The input features  
    y: The labels  batch_size: The desired batch size for the loader  
    shuffle: Boolean that determines if the data should be shuffled every epoch  
    type_conversion: The general required type conversion for different tasks,            
                    you can provide values as "classification" and "regression",           
                    otherwise, there is no type conversion.  
    """  
    # The provided X and y should be numpy arrays  
    # Inputs and labels  
    torch_inputs = None  
    torch_labels = None  
    if type_conversion == 'classification':    
        torch_inputs = torch.from_numpy(X).float()    
        torch_labels = torch.from_numpy(y).long()  
    elif type_conversion == 'regression':   
        torch_inputs = torch.from_numpy(X).float()    
        torch_labels = torch.from_numpy(y).float() 
    else:   
        torch_inputs = torch.from_numpy(X)
        torch_labels = torch.from_numpy(y)  
    # TensorDataset  
    torch_dataset = TensorDataset(torch_inputs, torch_labels) 
     # Generate the dataloader  
    torch_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)  
    
    return torch_loader
