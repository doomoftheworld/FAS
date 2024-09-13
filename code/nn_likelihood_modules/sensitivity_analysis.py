import copy
import numpy as np
from common_use_functions import most_common, softmax
from common_imports import tqdm

"""
Functions for sobol indices
"""
# Functions for sampling data and mixing data
def create_dataA(data, N, replace=True):
  """
  Sample rows from the 2D data array (data is in form of [number_of_data, number_of_features])
  """
  if N <= data.shape[0]:
    return copy.deepcopy(data[np.random.choice(data.shape[0], N, replace=replace), :])
  else:
    print('Please provide a valid number of sample size to sample data (smaller than the number of data)')
    return None

def create_dataB(data, N, replace=True):
  """
  Sample rows from the 2D data array (data is in form of [number_of_data, number_of_features])
  """
  if N <= data.shape[0]:
    return copy.deepcopy(data[np.random.choice(data.shape[0], N, replace=replace), :])
  else:
    print('Please provide a valid number of sample size to sample data (smaller than the number of data)')
    return None
  
def create_data_exclude_index(data, N, index_to_exclude, replace=True):
  """
  Sample rows from the 2D data array (data is in form of [number_of_data, number_of_features])
  and sample data only from the selected part of data (exclude the rows at "index_to_exclude").

  Note: If the rest of data are not enough for sampling, we still sample data from the whole provided data.
  """
  if N <= data.shape[0]:
    # Exclude the rows at the indicated indices
    rest_data = np.delete(copy.deepcopy(data), index_to_exclude, 0)
    # Decide how to return the sampled data (if the rest of data is not enough for the sample size, we still return samples from the whole data)
    if N <= rest_data.shape[0]:
      return copy.deepcopy(rest_data[np.random.choice(data.shape[0], N, replace=replace), :])
    else:
      return copy.deepcopy(data[np.random.choice(data.shape[0], N, replace=replace), :])
  else:
    print('Please provide a valid number of sample size to sample data (smaller than the number of data)')
    return None

def create_dataAB(dataA, dataB, variable_index_to_fix):
  """
  Build the mixed data
  """
  dataB_withA = copy.deepcopy(dataB)
  dataB_withA[:, variable_index_to_fix] = dataA[:, variable_index_to_fix]
  return dataB_withA

def generate_sample_index(data, N, replace=True):
  """
  Generate the index used to select samples (data is in form of [number_of_data, number_of_features])
  """
  if N <= data.shape[0]:
    return np.random.choice(data.shape[0], N, replace=replace)
  else:
    print('Please provide a valid number of sample size to get th sample indices (smaller than the number of data)')
    return None
  
def generate_sample_index_exclude_items(data, N, index_to_exclude, replace=True):
  """
  Generate the index used to select samples (data is in form of [number_of_data, number_of_features]),
  and sample indices only from the selected part of data (exclude "index_to_exclude").

  Note: If the rest of data are not enough for sampling, we still sample indices from the whole provided data.
  """
  if N <= data.shape[0]:
    # Determine the rest of index we could sample from
    rest_indices = np.delete(np.arange(data.shape[0]), index_to_exclude, 0)
    # Decide how to sample the indices
    if N <= rest_indices.shape[0]:
      return np.random.choice(rest_indices, N, replace=replace)
    else:
      return np.random.choice(data.shape[0], N, replace=replace)
  else:
    print('Please provide a valid number of sample size to get th sample indices (smaller than the number of data)')
    return None
  
def take_copy_of_data_by_index(data, index):
  """
  This function takes a copy of rows (i.e. data examples) at the indicated indices (i.e. parameter "index") from the provided numpy array.

  Note: index could be a list not just single index.
  """
  if index is not None:
    return copy.deepcopy(data[index, :])
  else:
    return None


"""
Function to get the evaluated values (function value y)
"""
def evaluate_y(model, data):
  """
  Now it is a static version without any pytorch adaptation, we could modify this later

  We consider now specially the model parameter as:
  {
    'weight' : ...,
    'bias' : ...,
  }
  where both weight and bias are numpy arrays

  This provided model variable should temporarily only be the parameters for the evaluated class

  This function calculates a simple linear transformation.
  """

  return np.dot(data, np.transpose(model['weight'])) + model['bias']

def evaluate_y_softmax(model, data):
  """
  Now it is a static version without any pytorch adaptation, we could modify this later

  We consider now specially the model parameter as:
  {
    'weight' : ...,
    'bias' : ...,
  }
  where both weight and bias are numpy arrays

  This provided model variable should temporarily only be the parameters for the evaluated class

  This function calculates a simple linear transformation with softmax activation.
  """

  return softmax(np.dot(data, np.transpose(model['weight'])) + model['bias'])

def build_per_class_linear_model(final_linear_params):
  """
  This is a temporary function that build quickly the needed linear model per class for the sobol indices evaluation.

  final_linear_params: The parameters (weight and bias) of the last linear prediction layer. 
  (it follows this structure:
  {
    'weight' : ... (numpy array containing weights of all classes),
    'bias' : ... (numpy array containing bias of all classes)
  }
  )
  """
  # The built parameter per class
  built_params_per_class = {}
  # Get the weight and bias
  final_linear_weight = final_linear_params['weight']
  final_linear_bias = final_linear_params['bias']
  # Build the parameter per class
  for classId in range(len(final_linear_bias)):
    # Assign the params
    built_params_per_class[classId] = {}
    built_params_per_class[classId]['weight'] = final_linear_weight[classId, :]
    built_params_per_class[classId]['bias'] = final_linear_bias[classId]

  return built_params_per_class

"""
Evaluate the sobel indices
"""
def sobol_indice_1st_and_total_order(model, variable_index, dataA, dataB, func_evaluate_y):
  """
  This function applies the sobol index calculation

  func_evaluate_y: The function you use to evaluate the y values.
  """

  dataB_withA = create_dataAB(dataA, dataB, variable_index)

  N = len(dataA)

  y_A = func_evaluate_y(model, dataA)
  y_AB = func_evaluate_y(model, dataB_withA)
  y_B = func_evaluate_y(model, dataB)

  num_1st_order = N*np.sum(np.multiply(y_A,y_AB)) - (np.sum(y_A)*np.sum(y_AB))
  num_tot = N*np.sum(np.multiply(y_B,y_AB)) - (np.sum(y_A)**2)
  deno = N*np.sum(y_A**2) - (np.sum(y_A))**2

  return np.round(num_1st_order/deno, 3), np.round((1 - (num_tot/deno)), 3)

"""
Functions that applies monte carlo sobol index experim
"""
def important_variable_by_sobol_index(data, model_params, nb_vars, func_evaluate_y, N=4096):
    """
    This function applies a complete sobol index experiment to get the important neurons
    
    data: The original dataset
    model_params: Model parameters (Here we consider the special case: provide the linear parameter of the last layer by numpy array)
    nb_vars: Number of variables to be evaluated
    func_evaluate_y: Function used to evaluate the Y value
    N: Sample size to sample the data
    """
#     ## Version 1
#     # Create different sets of data
#     dataA = create_dataA(data, N)
#     dataB = create_dataB(data, N)
    # Version 2 (Exclusif sampling)
    # Create different sets of data
    A_index = generate_sample_index(data, N, replace=False)
    B_index = generate_sample_index_exclude_items(data, N, A_index, replace=False)
    dataA = take_copy_of_data_by_index(data, A_index)
    dataB = take_copy_of_data_by_index(data, B_index)
    # The class list
    class_list = list(model_params.keys())
    # Calculate the sobol indices per class
    sobol_indices_per_class = {}
    for classId in class_list:
        # Get the current class parameters    
        current_class_params = model_params[classId]
        # Initialize the indices of the current class
        sobol_indices_per_class[classId] = {}
        # Evaluate the sobol indices     
        for var_index in tqdm(list(range(nb_vars)), desc='Processed variables of class ' + str(classId)):
            current_var_sobol_indices = sobol_indice_1st_and_total_order(current_class_params, var_index, dataA, dataB, func_evaluate_y)
            sobol_indices_per_class[classId][var_index] = current_var_sobol_indices
            
    ## Take only the first order sobol indices and apply essential statistics
    # Get the first and total order sobol indices
    first_order_sobol_indices_per_class = {}
    total_order_sobol_indices_per_class = {}
    for classId in sobol_indices_per_class:
        # Current class sobol indices
        current_class_sobol_indices = sobol_indices_per_class[classId]
        # Initialize the result and get the corresponding data     
        first_order_sobol_indices_per_class[classId] = {}
        total_order_sobol_indices_per_class[classId] = {}
        # Assign the first and total order sobol indices     
        for var_index in current_class_sobol_indices:
            first_order_sobol_indices_per_class[classId][var_index] = current_class_sobol_indices[var_index][0]
            total_order_sobol_indices_per_class[classId][var_index] = current_class_sobol_indices[var_index][1]
            
    ## Apply statistics
    # The most common index values
    most_common_first_order_indices_per_class = {}
    most_common_total_order_indices_per_class = {}
    average_first_order_indices_per_class = {}
    average_total_order_indices_per_class = {}
    for classId in class_list:
        # Take the first and total order index values
        first_order_indices_values = list(first_order_sobol_indices_per_class[classId].values())
        total_order_indices_values = list(total_order_sobol_indices_per_class[classId].values())
        # first order most common values
        most_common_first_order_indices_per_class[classId] = most_common(first_order_indices_values)
        # total order most common values
        most_common_total_order_indices_per_class[classId] = most_common(total_order_indices_values)
        # first order average values
        average_first_order_indices_per_class[classId] = np.mean(first_order_indices_values)
        # total order average values
        average_total_order_indices_per_class[classId] = np.mean(total_order_indices_values)

    # Determine the important variables
    important_variables_per_class = {}
    for classId in sobol_indices_per_class:
        important_variables_per_class[classId] = {}
        current_class_sobol_indices = sobol_indices_per_class[classId]
        for var_index in current_class_sobol_indices:
            first_order_sobol_index = current_class_sobol_indices[var_index][0]
            total_order_sobol_index = current_class_sobol_indices[var_index][1]
            if first_order_sobol_index > most_common_first_order_indices_per_class[classId] and  total_order_sobol_index > most_common_total_order_indices_per_class[classId]:
                important_variables_per_class[classId][var_index] = current_class_sobol_indices[var_index] 
                
    return important_variables_per_class

"""
Functions that applies important variable analysis based on sobol indices calculated by R
"""
def important_variables_R_first_order_analysis(R_first_order_sobol_per_class_dict, filter_method='mean'):
    """
    This function applies an anlysis with the calculated first order sobol indices from R sobol methods.

    R_first_order_sobol_per_class_dict: The calculated first order sobol indices with one R method, this param should be a dictionary.
    filter_method: The method used to determine the important variables, it could take two values: mean, median and most_common.
    """
    ## Apply statistics
    # The most common index values
    most_common_first_order_indices_per_class_R = {}
    mean_first_order_indices_per_class_R = {}
    median_first_order_indices_per_class_R = {}
    for classId in R_first_order_sobol_per_class_dict:
        # Take the current class sobol indices
        current_class_first_order_sobol_indices = list(R_first_order_sobol_per_class_dict[classId].values())
        # first order most common and mean values
        most_common_first_order_indices_per_class_R[classId] = most_common(current_class_first_order_sobol_indices)
        mean_first_order_indices_per_class_R[classId] = np.mean(current_class_first_order_sobol_indices)
        median_first_order_indices_per_class_R[classId] = np.median(current_class_first_order_sobol_indices)

    # Determine the important variables
    wrong_filter_method = False
    important_variables_per_class_R = {}
    for classId in R_first_order_sobol_per_class_dict:
        important_variables_per_class_R[classId] = {}
        current_class_first_order_sobol_indices_R = R_first_order_sobol_per_class_dict[classId]
        for var_index in current_class_first_order_sobol_indices_R:
            first_order_sobol_index_R = current_class_first_order_sobol_indices_R[var_index]
            if filter_method == 'most_common':
              if first_order_sobol_index_R > most_common_first_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = current_class_first_order_sobol_indices_R[var_index]
            elif filter_method == 'mean':
              if first_order_sobol_index_R > mean_first_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = current_class_first_order_sobol_indices_R[var_index]
            elif filter_method == 'median':
              if first_order_sobol_index_R > median_first_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = current_class_first_order_sobol_indices_R[var_index]
            else:
               wrong_filter_method = True
               break
            
    # Display if the provided filtering method is correct
    if wrong_filter_method:
       print("The provided filtering method is not correct, please provide a valid value between \"mean\" and \"most_common\", an empty dictionary is returned.")
                
    return important_variables_per_class_R


def important_variables_R_first_and_total_order_analysis(R_first_order_sobol_per_class_dict, R_total_order_sobol_per_class_dict, class_list, filter_method='mean', divide_factor=4):
    """
    This function applies an anlysis with the calculated first order sobol indices from R sobol methods.

    R_first_order_sobol_per_class_dict: The calculated first order sobol indices with one R method, this param should be a dictionary.
    R_total_order_sobol_per_class_dict: The calculated total order sobol indices with one R method, this param should be a dictionary.
    class_list: The list of classes of the considered classification problem
    filter_method: The method used to determine the important variables, it could take the possible values mentioned in the function.
    divide_factor: The factor used determine the threshold position. (e.g., 2 = median, 4 = first quarter)
    """
    ## Apply statistics
    # First order values
    most_common_first_order_indices_per_class_R = {}
    mean_first_order_indices_per_class_R = {}
    std_first_order_indices_per_class_R = {}
    median_first_order_indices_per_class_R = {}
    top_portion_first_order_indices_per_class_R = {}
    beside_end_portion_first_order_indices_per_class_R = {}
    # Total order values
    most_common_total_order_indices_per_class_R = {}
    mean_total_order_indices_per_class_R = {}
    std_total_order_indices_per_class_R = {}
    median_total_order_indices_per_class_R = {}
    top_portion_total_order_indices_per_class_R = {}
    beside_end_portion_total_order_indices_per_class_R = {}
    for classId in class_list:
        # Take the current class sobol indices
        current_class_first_order_sobol_indices = list(R_first_order_sobol_per_class_dict[classId].values())
        current_class_total_order_sobol_indices = list(R_total_order_sobol_per_class_dict[classId].values())
        # Get the number of examples in the first and total indices
        first_order_nb_examples = len(current_class_first_order_sobol_indices)
        total_order_nb_examples = len(current_class_total_order_sobol_indices)
        # Determine the position conversion term
        first_order_conv_term = 0 if first_order_nb_examples == 0 else -1
        total_order_conv_term = 0 if first_order_nb_examples == 0 else -1
        # Determine the desired "top portion" position index for the first and total order 
        # (the position conversion term is to convert it into index)
        first_order_top_portion_pos = int(first_order_nb_examples / divide_factor) + first_order_conv_term
        total_order_top_portion_pos = int(total_order_nb_examples / divide_factor) + total_order_conv_term
        # Determine the desired "beside end portion" position index for the first and total order 
        # (the position conversion term is to convert it into index)
        first_order_beside_end_portion_pos = first_order_nb_examples - int(first_order_nb_examples / divide_factor) + first_order_conv_term
        total_order_beside_end_portion_pos = total_order_nb_examples - int(total_order_nb_examples / divide_factor) + total_order_conv_term
        # Copy the array and sorted for top-portion position value determination
        copied_first_order_sobol_indices = copy.deepcopy(current_class_first_order_sobol_indices)
        copied_total_order_sobol_indices = copy.deepcopy(current_class_total_order_sobol_indices)
        copied_first_order_sobol_indices.sort(reverse=True)
        copied_total_order_sobol_indices.sort(reverse=True)
        # first order most common and mean values
        most_common_first_order_indices_per_class_R[classId] = most_common(current_class_first_order_sobol_indices)
        mean_first_order_indices_per_class_R[classId] = np.mean(current_class_first_order_sobol_indices)
        std_first_order_indices_per_class_R[classId] = np.std(current_class_first_order_sobol_indices)
        median_first_order_indices_per_class_R[classId] = np.median(current_class_first_order_sobol_indices)
        top_portion_first_order_indices_per_class_R[classId] = copied_first_order_sobol_indices[first_order_top_portion_pos]
        beside_end_portion_first_order_indices_per_class_R[classId] = copied_first_order_sobol_indices[first_order_beside_end_portion_pos]
        # total order most common and mean values
        most_common_total_order_indices_per_class_R[classId] = most_common(current_class_total_order_sobol_indices)
        mean_total_order_indices_per_class_R[classId] = np.mean(current_class_total_order_sobol_indices)
        std_total_order_indices_per_class_R[classId] = np.std(current_class_total_order_sobol_indices)
        median_total_order_indices_per_class_R[classId] = np.median(current_class_total_order_sobol_indices)
        top_portion_total_order_indices_per_class_R[classId] = copied_total_order_sobol_indices[total_order_top_portion_pos]
        beside_end_portion_total_order_indices_per_class_R[classId] = copied_total_order_sobol_indices[total_order_beside_end_portion_pos]

    # Get all the variable indices (we take the first class as an example to get the indices)
    var_indices = sorted(list(R_first_order_sobol_per_class_dict[class_list[0]].keys()))

    # Determine the important variables
    wrong_filter_method = False
    important_variables_per_class_R = {}
    for classId in class_list:
        important_variables_per_class_R[classId] = {}
        current_class_first_order_sobol_indices_R = R_first_order_sobol_per_class_dict[classId]
        current_class_total_order_sobol_indices_R = R_total_order_sobol_per_class_dict[classId]
        for var_index in var_indices:
            first_order_sobol_index_R = current_class_first_order_sobol_indices_R[var_index]
            total_order_sobol_index_R = current_class_total_order_sobol_indices_R[var_index]
            if filter_method == 'most_common':
              if first_order_sobol_index_R > most_common_first_order_indices_per_class_R[classId] and total_order_sobol_index_R > most_common_total_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
            elif filter_method == 'mean':
              if first_order_sobol_index_R > mean_first_order_indices_per_class_R[classId] and total_order_sobol_index_R > mean_total_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
            elif filter_method == 'median':
              if first_order_sobol_index_R > median_first_order_indices_per_class_R[classId] and total_order_sobol_index_R > median_total_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
            elif filter_method == 'top_portion':
              if first_order_sobol_index_R > top_portion_first_order_indices_per_class_R[classId] and total_order_sobol_index_R > top_portion_total_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
            elif filter_method == 'beside_end_portion':
              if first_order_sobol_index_R > beside_end_portion_first_order_indices_per_class_R[classId] and total_order_sobol_index_R > beside_end_portion_total_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
            elif filter_method == 'only_first_order_mean':
              if first_order_sobol_index_R > mean_first_order_indices_per_class_R[classId]:
                  important_variables_per_class_R[classId][var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
            elif filter_method == 'half_std_below_mean':
              first_order_sobol_threshold = mean_first_order_indices_per_class_R[classId] - 0.5 * std_first_order_indices_per_class_R[classId]
              total_order_sobol_threshold = mean_total_order_indices_per_class_R[classId] - 0.5 * std_total_order_indices_per_class_R[classId]
              if first_order_sobol_index_R > first_order_sobol_threshold and total_order_sobol_index_R > total_order_sobol_threshold:
                  important_variables_per_class_R[classId][var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
            else:
               wrong_filter_method = True
               break
            
    # Display if the provided filtering method is correct
    if wrong_filter_method:
       print("The provided filtering method is not correct, please provide a valid value between \"mean\" and \"most_common\", an empty dictionary is returned.")
                
    return important_variables_per_class_R


def important_variables_R_first_and_total_order_analysis_multout_ver(R_first_order_sobol_dict, R_total_order_sobol_dict, filter_method='mean', divide_factor=4):
    """
    This function applies an anlysis with the calculated first order sobol indices from R sobol methods.

    R_first_order_sobol_dict: The calculated first order sobol indices with one R method, this param should be a dictionary.
    R_total_order_sobol_dict: The calculated total order sobol indices with one R method, this param should be a dictionary.
    class_list: The list of classes of the considered classification problem
    filter_method: The method used to determine the important variables, it could take the possible values mentioned in the function.
    divide_factor: The factor used determine the threshold position. (e.g., 2 = median, 4 = first quarter)
    """
    ## Apply statistics
    # Take the current class sobol indices
    first_order_sobol_indices = list(R_first_order_sobol_dict.values())
    total_order_sobol_indices = list(R_total_order_sobol_dict.values())
    # Get the number of examples in the first and total indices
    first_order_nb_examples = len(first_order_sobol_indices)
    total_order_nb_examples = len(total_order_sobol_indices)
    # Determine the position conversion term
    first_order_conv_term = 0 if first_order_nb_examples == 0 else -1
    total_order_conv_term = 0 if first_order_nb_examples == 0 else -1
    # Determine the desired "top portion" position index for the first and total order 
    # (the position conversion term is to convert it into index)
    first_order_top_portion_pos = int(first_order_nb_examples / divide_factor) + first_order_conv_term
    total_order_top_portion_pos = int(total_order_nb_examples / divide_factor) + total_order_conv_term
    # Determine the desired "beside end portion" position index for the first and total order 
    # (the position conversion term is to convert it into index)
    first_order_beside_end_portion_pos = first_order_nb_examples - int(first_order_nb_examples / divide_factor) + first_order_conv_term
    total_order_beside_end_portion_pos = total_order_nb_examples - int(total_order_nb_examples / divide_factor) + total_order_conv_term
    # Copy the array and sorted for top-portion position value determination
    copied_first_order_sobol_indices = copy.deepcopy(first_order_sobol_indices)
    copied_total_order_sobol_indices = copy.deepcopy(total_order_sobol_indices)
    copied_first_order_sobol_indices.sort(reverse=True)
    copied_total_order_sobol_indices.sort(reverse=True)
    # first order most common and mean values
    most_common_first_order_indices_R = most_common(first_order_sobol_indices)
    mean_first_order_indices_R = np.mean(first_order_sobol_indices)
    std_first_order_indices_R = np.std(first_order_sobol_indices)
    median_first_order_indices_R = np.median(first_order_sobol_indices)
    top_portion_first_order_indices_R = copied_first_order_sobol_indices[first_order_top_portion_pos]
    beside_end_portion_first_order_indices_R = copied_first_order_sobol_indices[first_order_beside_end_portion_pos]
    # total order most common and mean values
    most_common_total_order_indices_R = most_common(total_order_sobol_indices)
    mean_total_order_indices_R = np.mean(total_order_sobol_indices)
    std_total_order_indices_R = np.std(total_order_sobol_indices)
    median_total_order_indices_R = np.median(total_order_sobol_indices)
    top_portion_total_order_indices_R = copied_total_order_sobol_indices[total_order_top_portion_pos]
    beside_end_portion_total_order_indices_R = copied_total_order_sobol_indices[total_order_beside_end_portion_pos]

    # Get all the variable indices (we take the first class as an example to get the indices)
    var_indices = sorted(list(R_first_order_sobol_dict.keys()))

    # Determine the important variables
    wrong_filter_method = False
    important_variables_R = {}
    for var_index in var_indices:
        first_order_sobol_index_R = first_order_sobol_indices[var_index]
        total_order_sobol_index_R = total_order_sobol_indices[var_index]
        if filter_method == 'most_common':
          if first_order_sobol_index_R > most_common_first_order_indices_R and total_order_sobol_index_R > most_common_total_order_indices_R:
              important_variables_R[var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
        elif filter_method == 'mean':
          if first_order_sobol_index_R > mean_first_order_indices_R and total_order_sobol_index_R > mean_total_order_indices_R:
              important_variables_R[var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
        elif filter_method == 'median':
          if first_order_sobol_index_R > median_first_order_indices_R and total_order_sobol_index_R > median_total_order_indices_R:
              important_variables_R[var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
        elif filter_method == 'top_portion':
          if first_order_sobol_index_R > top_portion_first_order_indices_R and total_order_sobol_index_R > top_portion_total_order_indices_R:
              important_variables_R[var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
        elif filter_method == 'beside_end_portion':
          if first_order_sobol_index_R > beside_end_portion_first_order_indices_R and total_order_sobol_index_R > beside_end_portion_total_order_indices_R:
              important_variables_R[var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
        elif filter_method == 'only_first_order_mean':
          if first_order_sobol_index_R > mean_first_order_indices_R:
              important_variables_R[var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
        elif filter_method == 'half_std_below_mean':
          first_order_sobol_threshold = mean_first_order_indices_R - 0.5 * std_first_order_indices_R
          total_order_sobol_threshold = mean_total_order_indices_R - 0.5 * std_total_order_indices_R
          if first_order_sobol_index_R > first_order_sobol_threshold and total_order_sobol_index_R > total_order_sobol_threshold:
              important_variables_R[var_index] = (first_order_sobol_index_R, total_order_sobol_index_R)
        else:
            wrong_filter_method = True
            break
            
    # Display if the provided filtering method is correct
    if wrong_filter_method:
       print("The provided filtering method is not correct, please provide a valid value between \"mean\" and \"most_common\", an empty dictionary is returned.")
                
    return important_variables_R