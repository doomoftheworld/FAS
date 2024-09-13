"""
Function for the activation levels distribution calculation
"""

import warnings
from common_use_functions import get_actual_time, create_directory, read_csv_to_pd_df
from common_imports import *

"""
Functions of the required calculations for the distribution generation
"""
# Distribution calculation and conservation
def binId_calc(value, step):
    if step == 0:
        if value >= 0:
            return int(value)
        else:
            # Temporarilly, we choose to exit
            print("Negative value of activation level appered, exiting the program.")
            exit(3)
    else:
        if value >= 0:
            return int(value / step)
        else:
            # Temporarilly, we choose to exit
            print("Negative value of activation level appered, exiting the program.")
            exit(3)
#             return int(value / step)-1

def distribution_calculation_neuron_ver(set_class, layer, actLevel, set_name, model_name, step_coeff=1, display_warn=False):
    to_compared_little_value = 0.000001 # 1e-6
    distribution_array = []
    actLevel_neuron_as_line = actLevel.transpose()
    nb_neuron = actLevel_neuron_as_line.shape[0]
    nb_elem = actLevel_neuron_as_line.shape[1]
    # The vectorized binId function has been moved out of the for loop in order to avoid multiple intialization
    binId_func = np.vectorize(binId_calc)
    for i in range(nb_neuron):
        mean = np.mean(actLevel_neuron_as_line[i])
        std_dev = np.std(actLevel_neuron_as_line[i])
        if std_dev < to_compared_little_value and display_warn:
            warnings.warn(model_name + ": standard deviation too small at the neuron " + str(i) + " of layer " + str(layer) + " for the " + set_name + ", please be aware.")
        step = std_dev*step_coeff
        binId = binId_func(actLevel_neuron_as_line[i], step)
        uniq_binId, bin_counts = np.unique(binId, return_counts=True)
        for j in range(len(uniq_binId)):
            distribution_array.append([layer, i, set_class, uniq_binId[j],
                                       uniq_binId[j]*step, (uniq_binId[j]+1)*step, bin_counts[j]])  
    return distribution_array

def distribution_calculation_whole_ver(set_class, layer, actLevel, set_name, model_name, step_coeff=1, display_warn=False):
    to_compared_little_value = 0.000001 # 1e-6
    distribution_array = []
    actLevel_neuron_as_line = actLevel.transpose()
    nb_neuron = actLevel_neuron_as_line.shape[0]
    nb_elem = actLevel_neuron_as_line.shape[1]
    mean = np.mean(actLevel_neuron_as_line, axis=1, keepdims=True)
    std_dev = np.std(actLevel_neuron_as_line, axis=1, keepdims=True)
    # The vectorized binId function has been moved out of the for loop in order to avoid multiple intialization
    binId_func = np.vectorize(binId_calc)
    for i in range(nb_neuron):
        if std_dev[i][0] < to_compared_little_value and display_warn:
            warnings.warn(model_name + ": standard deviation too small at the neuron " + str(i) + " of layer " + str(layer) + " for the " + set_name + ", please be aware.")
        step = std_dev[i][0]*step_coeff
        binId = binId_func(actLevel_neuron_as_line[i], step)
        uniq_binId, bin_counts = np.unique(binId, return_counts=True)
        for j in range(len(uniq_binId)):
            distribution_array.append([layer, i, set_class, uniq_binId[j],
                                       uniq_binId[j]*step, (uniq_binId[j]+1)*step, bin_counts[j]])            
    return distribution_array

"""
Distribution generation
"""
def generate_distributions(registered_actLevel, set_name, model_name):
    distributions = []
    
    entry_class = registered_actLevel['class']
    uniq_entry_class = np.unique(entry_class)
    entry_index_by_class = {}
    for class_value in uniq_entry_class:
        index_for_class,_ = np.where(entry_class == class_value)
        entry_index_by_class[class_value] = index_for_class
    
    for layer, actLevel in registered_actLevel['actLevel'].items():
        for class_value, class_index in entry_index_by_class.items():
            class_layer_distribution = distribution_calculation_neuron_ver(class_value, layer, actLevel[class_index], set_name, model_name)
            for neuron_info in class_layer_distribution:
                distributions.append(neuron_info)
    
    return distributions

"""
Save the generated distributions 
"""    
def save_distributions_after_training(current_path, model_name, train_distributions, test_distributions):
    # dd/mm/YY H:M:S
    dt_string = get_actual_time()
    
    # Create directory
    distributions_dir_mame = current_path + '\\distributions_' +  model_name + '_' + dt_string
    create_directory(distributions_dir_mame)
    
    # Distribution headers
    headers = ['layerId', 'nodeId', 'classId', 'binId', 'sigmaInterval_lb', 'sigmaInterval_ub', 'sigmaFreq']
    
    # Write train set data
    print('Starting the writing of the train set distributions...')
    csv_train_distribution_name = distributions_dir_mame + '\\distribution_train_set.csv'
    pd.DataFrame(train_distributions, columns=headers).to_csv(csv_train_distribution_name,index=False, sep=' ')
    
    # Write test set data
    print('Starting the writing of the test set distributions...')
    csv_test_distribution_name = distributions_dir_mame + '\\distribution_test_set.csv'
    pd.DataFrame(test_distributions, columns=headers).to_csv(csv_test_distribution_name,index=False, sep=' ')
    
    print('The distributions of the train set and the test set are saved.')
    
def save_distributions(folder_path, distributions, set_name):  
    # Distribution headers
    headers = ['layerId', 'nodeId', 'classId', 'binId', 'sigmaInterval_lb', 'sigmaInterval_ub', 'sigmaFreq']
    
    # Write set data
    print('Starting the writing of the ' + set_name + ' set distributions...')
    csv_distribution_name = folder_path + '\\distribution_' + set_name + '_set.csv'
    pd.DataFrame(distributions, columns=headers).to_csv(csv_distribution_name,index=False, sep=' ')
    
    print('The distribution of the ' + set_name + ' set is saved.')
    
    return csv_distribution_name


"""
Two tasks (generate and save) together for the calculated distribution of the activation levels obtained from the model after the training
"""
def generate_and_save_distributions_after_training(current_path, model_name, train_actLevel, test_actLevel):
    # dd/mm/YY H:M:S
    dt_string = get_actual_time()
    
    # Create directory
    distributions_dir_mame = current_path + '\\distributions_' +  model_name + '_' + dt_string
    create_directory(distributions_dir_mame)
    
    # Distribution headers
    headers = ['layerId', 'nodeId', 'classId', 'binId', 'sigmaInterval_lb', 'sigmaInterval_ub', 'sigmaFreq']
    
    # Write train set data
    print('Starting the writing of the train set distributions...')
    train_distributions = generate_distributions(train_actLevel, 'train set')
    csv_train_distribution_name = distributions_dir_mame + '\\distribution_train_set.csv'
    pd.DataFrame(train_distributions, columns=headers).to_csv(csv_train_distribution_name,index=False, sep=' ')
    
    # Write test set data
    print('Starting the writing of the test set distributions...')
    test_distributions = generate_distributions(test_actLevel, 'test set')
    csv_test_distribution_name = distributions_dir_mame + '\\distribution_test_set.csv'
    pd.DataFrame(test_distributions, columns=headers).to_csv(csv_test_distribution_name,index=False, sep=' ')
    
    print('The distributions of the train set and the test set are saved.')


def generate_and_save_distributions_of_history(current_path, history):
    # dd/mm/YY H:M:S
    dt_string = get_actual_time()
    
    # Create directory
    distributions_dir_mame = current_path + '\\distributions_' +  history['model_name'] + '_' + dt_string
    create_directory(distributions_dir_mame)
    
    # Distribution headers
    headers = ['layerId', 'nodeId', 'classId', 'binId', 'sigmaInterval_lb', 'sigmaInterval_ub', 'sigmaFreq']
       
    # Writing every epoch's history
    for epoch, registered_history in history['training'].items():
        print('Starting the writing of the distributions for epoch ' + str(epoch) + '...')
        epoch_dir_name = distributions_dir_mame + '\\epoch_' + str(epoch + 1)
        create_directory(epoch_dir_name)
        
        # Write during training data
        print('Starting the writing of the distributions generated during training...')
        during_training_distributions = generate_distributions(registered_history['during_training'], 'set of the process during training')
        csv_during_training_distribution_name = epoch_dir_name + '\\distribution_during_training.csv'
        pd.DataFrame(during_training_distributions, columns=headers).to_csv(csv_during_training_distribution_name,index=False, sep=' ')
            
        # Write train set data
        print('Starting the writing of the train set distributions...')
        train_distributions = generate_distributions(registered_history['train'], 'train set')
        csv_train_distribution_name = epoch_dir_name + '\\distribution_train_set.csv'
        pd.DataFrame(train_distributions, columns=headers).to_csv(csv_train_distribution_name,index=False, sep=' ')
                
        # Write validation set data
        print('Starting the writing of the valid set distributions...')
        valid_distributions = generate_distributions(registered_history['valid'], 'valid set')
        csv_valid_distribution_name = epoch_dir_name + '\\distribution_valid_set.csv'
        pd.DataFrame(valid_distributions, columns=headers).to_csv(csv_valid_distribution_name,index=False, sep=' ')
    
    # Writing test history
    print('Starting the writing of the test set distributions...')
    test_dir_name = distributions_dir_mame + '\\test'
    create_directory(test_dir_name)
    test_distributions = generate_distributions(history['test'], 'test set')
    csv_test_distribution_name = test_dir_name + '\\distribution_test_set.csv'
    pd.DataFrame(test_distributions, columns=headers).to_csv(csv_test_distribution_name,index=False, sep=' ')
    
    print('All distributions saved.')


