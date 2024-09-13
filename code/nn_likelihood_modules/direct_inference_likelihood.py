from common_use_functions import path_join, save_content_to_file, create_directory, erase_files_from_folder, save_list_to_csv_without_np_conversion, build_map_to_index_dict, save_df_to_csv, erase_one_file
from common_imports import *
from constant import *
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from subprocess import PIPE, run
from vector_preprocessing import get_mean_std
from scipy.stats import wasserstein_distance

"""
Activation levels to the required form of data files
"""
def convert_entry_data_to_likelihood_file_np_ver(entry_data, layerId, entryId, save_path, display_process=False, ret_path=False):
    """
    Convert 1D numpy array which contains the activation levels to the required file form
    """
    headers = ['layerId', 'nodeId', 'nodeContrib']
    converted_list = []
    for nodeId, nodeContrib in enumerate(entry_data.tolist()):
        converted_list.append([layerId, nodeId, nodeContrib])
    converted_filename = 'img_' + str(entryId) + '_nodeContribs.csv'
    converted_file_path = path_join(save_path, converted_filename)
    save_list_to_csv_without_np_conversion(converted_file_path, converted_list, headers=headers, sep=' ')
    if display_process:
        print('Done saving for entry', entryId)
    if ret_path:
        return (converted_filename, converted_file_path)
    else:
        return converted_filename

def activation_levels_to_required_data_files(layerId, layer_actLevels, layer_save_path, display=True):
    """
    display: The parameter whcih controls if we would like to display the execution process
    """
    if display:
        print('Converting activation levels of layer', str(layerId), 'to the required data files...')
    converted_files = []
    layer_actLevels_iterator = tqdm(layer_actLevels, desc='Converted entries:') if display else layer_actLevels
    for index, actLevel in enumerate(layer_actLevels_iterator):
        # File building
        entry_filename = convert_entry_data_to_likelihood_file_np_ver(actLevel, layerId, index, layer_save_path)
        converted_files.append(entry_filename)
    return converted_files

"""
Likelihood experiment preparation function
"""
def make_dat_filename(name):
    return name + dat_extension


"""
Likelihood execution
"""
def likelihood_result_process(results, layerId, with_classId=True):
    # layerId is an additional information to create the desired data
    lines = results.split('\n')
    # Variable to store the results
    processed_likelihoods = []
    # Simple Version
    for line in lines:
        if 'TAB' in line:
            used_results = line.split(' ')
            entryId = int(used_results[-3])
            classId = int(used_results[-2])
            dist_result = float(used_results[-1])
            if with_classId:
                processed_likelihoods.append([layerId, classId, entryId, dist_result])
            else:
                processed_likelihoods.append([layerId, entryId, dist_result])
    return processed_likelihoods
   

def exec_likelihood(data_path, hist_prob_path, data_list_file_path, classId, layerId, display=False, with_classId=True, use_absolute_module_path=False):
    """
    use_absolute_module_path: Use the absolute module path to access the likelihood distance calculation module from Ettore 

    Note: If you want to use the absolte path, please configure it in the file "constant.py"
    """
    exec_path = None
    if use_absolute_module_path:
        exec_path = path_join(nn_likelihood_module_absolute_path, single_class_likelihood_path)
    else:
        exec_path = single_class_likelihood_path
    command = [python_command, exec_path, hist_prob_path, data_path, data_list_file_path, str(classId)]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    if result.returncode != 0:
        print('Error during the likelihood calculation.')
        print(result.stdout)
        print(result.stderr)
    else:
        if display:
            print(result.stdout)
    processed_result = likelihood_result_process(result.stdout, layerId, with_classId=with_classId)
    return processed_result

def all_files_likelihood_calc(files, data_path, hist_prob_path, entry_class, layerId, with_predict=True, pool_ver=True, nb_divide_task=4, use_absolute_module_path=False):
    likelihood_result = []
    exec_likelihood_option_params = None
    if with_predict:
        if use_absolute_module_path:
            exec_likelihood_option_params = (False, False, True)
        else:
            exec_likelihood_option_params = (False, False, False)
    else:
        if use_absolute_module_path:
            exec_likelihood_option_params = (False, True, True)
        else:
            exec_likelihood_option_params = (False, True, False)
    if pool_ver:
        # Divide the files to executed in nb_thread parts and save it to the corresponding "dataFileListindex.dat"
        dat_files_to_execute = []
        batch_size = int(len(files)/nb_divide_task)
        if batch_size == 0:
            batch_size = 1
            nb_divide_task = len(files)
        for index in range(nb_divide_task):
            current_files = None
            if index == nb_divide_task-1:
                current_files = files[index*batch_size:]
            else:
                current_files = files[index*batch_size:(index+1)*batch_size]
            current_data_list_filename = make_dat_filename(data_list_name+'_'+str(index))
            current_data_list_file_path = path_join(temp_path, current_data_list_filename)
            dat_files_to_execute.append(current_data_list_file_path)
            save_content_to_file(current_files, current_data_list_file_path)
        # Parrallel execution of the likelihood calculation for the divided parts (Old version: executor = ThreadPoolExecutor())
        with ThreadPoolExecutor() as executor:
            all_tasks = [executor.submit(lambda p : exec_likelihood(*p), (data_path, hist_prob_path, dat_file, entry_class, layerId, *exec_likelihood_option_params)) for dat_file in dat_files_to_execute]
            wait(all_tasks, return_when=ALL_COMPLETED)
            for task in all_tasks:
                task_result = task.result()
                likelihood_result.extend(task_result)
    else:
        data_list_file_path = path_join(temp_path, make_dat_filename(data_list_name))
        save_content_to_file(files, data_list_file_path)
        likelihood_result = exec_likelihood(data_path, hist_prob_path, data_list_file_path, entry_class, layerId, *exec_likelihood_option_params)
    return likelihood_result

def layer_whole_likelihood_experim(registered_actLevel, layerId, hist_prob_path, class_list, use_absolute_module_path=False, display=True):
    """
    Whole likelihood experiment of the activation levels at one layer

    display: Boolean indicating if we would like to display the execution process

    Note: You can use the “use_absolute_module_path” paramter to solve the question of relative path to the desired executable.
    However, you should always provid the "options.dat" file in your current working directory. (an example of this file is provided in the module) 
    """
    ## Experiment preparation
    # Take the corresponding data
    layer_actLevels = registered_actLevel['actLevel'][layerId]
    ground_truths = registered_actLevel['class'].reshape(-1)
    predicted_classes = registered_actLevel['predict_class'].reshape(-1)
    # predicted_classes = np.argmax(registered_actLevel['prob'], axis=1) # Without predicted class version
    # Create the temp folder
    create_directory(temp_path, display=display)
    # Create the folder to store the converted data
    layer_temp_path = path_join(temp_path, str(layerId))
    create_directory(layer_temp_path, display=display)
    # Convert the data
    converted_files = activation_levels_to_required_data_files(layerId, layer_actLevels, layer_temp_path, display=display)
    ## Build the whole likelihood
    # Evaluate likelihood per class
    layer_set_classes_likelihood = {}
    class_list_iterator = tqdm(class_list, desc='Processed class for the likelihood calculation: ') if display else class_list
    for entry_class in class_list_iterator:
        likelihood_result = all_files_likelihood_calc(converted_files, layer_temp_path, hist_prob_path, entry_class, layerId, with_predict=True, pool_ver=True, use_absolute_module_path=use_absolute_module_path)
        # Create a temporary dataframe stores the likelihood results for the current class
        class_likelihood_df = pd.DataFrame(likelihood_result, columns=distance_with_predict_class_file_headers[:3])
        layer_set_classes_likelihood[entry_class] = class_likelihood_df
    # Build the whole likelihood dataframe
    final_whole_distance_headers = copy.deepcopy(whole_distance_headers)
    final_whole_distance_headers.extend([dist_column+'_'+str(int(entry_class)) for entry_class in class_list])
    layer_whole_distances = pd.DataFrame(columns=final_whole_distance_headers)
    for index, entry_class in enumerate(layer_set_classes_likelihood.keys()):
        current_class_likelihood_df = layer_set_classes_likelihood[entry_class]
        layer_whole_distances[dist_column+'_'+str(int(entry_class))] = current_class_likelihood_df[dist_column].tolist()
        if index == 0:
            layer_whole_distances[final_whole_distance_headers[0]] = current_class_likelihood_df[final_whole_distance_headers[0]].tolist()
            layer_whole_distances[final_whole_distance_headers[1]] = current_class_likelihood_df[final_whole_distance_headers[1]].tolist()
    # Add the class related information
    layer_whole_distances[class_column] = ground_truths
    layer_whole_distances[predicted_class_column] = predicted_classes
    # Erase the content in the temp folder
    erase_files_from_folder(temp_path)
    return layer_whole_distances

def build_layer_train_set_infos(train_layer_whole_distances, class_list):
    """
    This function builds the statistical information about the distances of the training set inputs
    """
    ## Calculate the information and save it
    # Information calculation
    class_dist_infos = {}
    for entry_class in class_list:
        # Intialization
        class_dist_infos[str(entry_class)] = {}
        # Calculation     
        current_class_dist_col = dist_column+'_'+str(int(entry_class))
        current_class_entries =  train_layer_whole_distances[train_layer_whole_distances[class_column] == entry_class]
        current_class_dists = current_class_entries[current_class_dist_col].to_numpy()
        current_class_dist_mean,current_class_dist_std  = get_mean_std(current_class_dists)
        # Register the information
        class_dist_infos[str(int(entry_class))][mean_name] = current_class_dist_mean
        class_dist_infos[str(int(entry_class))][std_name] = current_class_dist_std
    return class_dist_infos

def map_to_predicted_class_distance(layer_whole_distances, class_list):
    """
    This function create a new dataframe by creating a new column named "dist" which has the corresponding distance of the predicted class
    """
    copied_distances_df = layer_whole_distances.copy(deep=True)
    copied_distances_df[dist_column] = copied_distances_df.apply(lambda row: row[dist_column+'_'+str(int(row[predicted_class_column]))], axis=1)
    copied_distances_df = copied_distances_df.drop(columns=[dist_column+'_'+str(int(entry_class)) for entry_class in class_list])
    return copied_distances_df

def filter_decision_based_on_train_infos(layer_whole_distances, class_dist_infos, std_threshold_coeff=1, original_class=False):
    """
    This function finds the index of the odd decisions based on the likelihood distance

    original_class: Boolean parameter which determines if we are filetring the entries based on the original class(Case: True) or the predicted class(Case: False)
    """
    filtered_index = []
    for index, row in layer_whole_distances.iterrows():
        current_ref_class = None
        if original_class:
            current_ref_class = row[class_column]
        else:
            current_ref_class = row[predicted_class_column]
        current_dist_column = dist_column + '_' + str(int(current_ref_class))
        current_entry_dist = row[current_dist_column]
        current_class_mean = class_dist_infos[str(int(current_ref_class))][mean_name]
        current_class_std = class_dist_infos[str(int(current_ref_class))][std_name]
        if current_entry_dist > current_class_mean + std_threshold_coeff*current_class_std:
            filtered_index.append(index)
    return filtered_index

def layer_one_class_likelihood_experim(registered_actLevel, layerId, hist_prob_path, single_class, use_absolute_module_path=False, display=True):
    """
    Single class likelihood experiment of the activation levels at one layer (i.e. according to the indicated "single class")

    display: Boolean indicating if we would like to display the execution process

    Note: You can use the “use_absolute_module_path” paramter to solve the question of relative path to the desired executable.
    However, you should always provid the "options.dat" file in your current working directory. (an example of this file is provided in the module) 
    """
    ## Experiment preparation
    # Take the corresponding data
    layer_actLevels = registered_actLevel['actLevel'][layerId]
    ground_truths = registered_actLevel['class'].reshape(-1)
    predicted_classes = registered_actLevel['predict_class'].reshape(-1)
    # predicted_classes = np.argmax(registered_actLevel['prob'], axis=1) # Without predicted class version
    # Create the temp folder
    create_directory(temp_path, display=display)
    # Create the folder to store the converted data
    layer_temp_path = path_join(temp_path, str(layerId))
    create_directory(layer_temp_path, display=display)
    # Convert the data
    converted_files = activation_levels_to_required_data_files(layerId, layer_actLevels, layer_temp_path, display=display)
    ## Build the single class likelihood
    # Evaluate likelihood of the desired class
    likelihood_result = all_files_likelihood_calc(converted_files, layer_temp_path, hist_prob_path, single_class, layerId, with_predict=True, pool_ver=True, use_absolute_module_path=use_absolute_module_path)
    # Create a temporary dataframe stores the likelihood results for the current class
    class_likelihood_df = pd.DataFrame(likelihood_result, columns=distance_with_predict_class_file_headers[:3])
    # Build the final likelihood dataframe
    final_distance_headers = copy.deepcopy(whole_distance_headers)
    final_distance_headers.extend([dist_column+'_'+str(int(single_class))])
    layer_single_class_distances = pd.DataFrame(columns=final_distance_headers)
    layer_single_class_distances[dist_column+'_'+str(int(single_class))] = class_likelihood_df[dist_column].tolist()
    layer_single_class_distances[final_distance_headers[0]] = class_likelihood_df[final_distance_headers[0]].tolist()
    layer_single_class_distances[final_distance_headers[1]] = class_likelihood_df[final_distance_headers[1]].tolist()
    # Add the class related information
    layer_single_class_distances[class_column] = ground_truths
    layer_single_class_distances[predicted_class_column] = predicted_classes
    # Erase the content in the temp folder
    erase_files_from_folder(temp_path)
    return layer_single_class_distances

"""
Filtering (OOD detection) quality evaluation according to accuracy
"""
def evaluate_filtering(whole_distances, filtered_index, set_name, display=True):
    """
    This function evaluates the quality of the OOD detection (filtering).
    
    whole_distances: The calculated whole likelihood distances
    filtered_index: The determined filtering indices (it should be coherent with the provided whole likelihood distances)
    set_name: The set name on which we are doing evaluation
    """
    ## Display the detailed accuracy information
    # Evaluate the original accuracy
    acc = np.sum(whole_distances[class_column]==whole_distances[predicted_class_column]) / whole_distances.shape[0]
    # Get the filtered images
    filtered_distances = whole_distances.loc[whole_distances.index.isin(filtered_index)]
    non_filtered_distances = whole_distances[~whole_distances.index.isin(filtered_distances.index)]
    # Accuracy on the filtered images
    filtered_acc = None
    if filtered_distances.shape[0] != 0:
        filtered_acc = np.sum(filtered_distances[class_column]==filtered_distances[predicted_class_column]) / filtered_distances.shape[0]
    else:
        filtered_acc = -1
    # Accuracy on the non-filtered images
    non_filtered_acc = None
    if non_filtered_distances.shape[0] != 0:
        non_filtered_acc = np.sum(non_filtered_distances[class_column]==non_filtered_distances[predicted_class_column]) / non_filtered_distances.shape[0]
    else:
        non_filtered_acc = -1
    if display:
        # Display the number of examples
        print("The number of all the", set_name, "examples :", whole_distances.shape[0])
        print("The number of filtered", set_name, "examples:", filtered_distances.shape[0])
        print("The number of non-filtered", set_name, "examples :", non_filtered_distances.shape[0])
        # Jump line between two displays
        print()
        # Display the accuracy
        print("The accuracy on all the", set_name, "examples :", acc)
        print("The accuracy on the filtered", set_name, "examples :", filtered_acc)
        print("The accuracy on the non-filtered", set_name, "examples :", non_filtered_acc)

    return whole_distances.shape[0], filtered_distances.shape[0], non_filtered_distances.shape[0], acc, filtered_acc, non_filtered_acc

def wasserstein_distrib_distance(distrib_one_values, distrib_two_values):
    """
    This function calculates the wasserstein distance between two distributions

    distrib_one_values: The values of the first distribution
    distrib_two_values: The values of the second distribution
    """
    return wasserstein_distance(distrib_one_values, distrib_two_values)

"""
The following functions are used for the post-processing after finding the sobol indices
"""
def build_actLevel_important_vars(actLevels, sorted_important_var_by_class, layerId):
    """
    This function build the activation levels of the desired layer in a way that they only contain the important variables
    
    actLevels: The original activation levels
    sorted_important_var_by_class: The sorted important variables at the desired layer
    layerId: The desired layer to be built (the chosen layer should be coherent with the provided important variables)
    """
    # Initialize the built activation levels by class     
    built_actLevels_by_class = {}
    for classId in sorted_important_var_by_class:
        # Get the important vars
        current_important_var_indices = sorted_important_var_by_class[classId]
        # Distorted image activation levels     
        actLevels_of_important_vars = copy.deepcopy(actLevels)
        actLevels_of_important_vars['actLevel'][layerId] = actLevels_of_important_vars['actLevel'][layerId][:, current_important_var_indices]
        built_actLevels_by_class[classId] = actLevels_of_important_vars
        
    return built_actLevels_by_class

def calculate_likelihood_with_important_vars(distribution_folder_path, built_actLevels_by_class,
                                             layerId, distrib_prefix='distribution_important_var_class_'):
    """
    This function executes the experiment to have the likelihood distances
    with only the important variables for different classes.
    
    distribution_folder_path: The path of the folder that contains different distributions of each class.
    built_actLevels_by_class: The built activation levels by class with only the important variables on the layer to be evaluated
    layerId: The desired layer to be evaluated
    distrib_prefix: The prefix of the distribution files
    """
    # Generate the likelihood of each class with only the important variables     
    likelihood_by_class = {}
    for classId in built_actLevels_by_class:
        current_class_distrib_file_path = path_join(distribution_folder_path, distrib_prefix+str(classId)+'.csv')
        likelihood_by_class[classId] = layer_one_class_likelihood_experim(built_actLevels_by_class[classId], layerId, 
                                                                                      current_class_distrib_file_path, classId,
                                                                                      use_absolute_module_path=True)
    # Merge the likelihood from different class
    merged_whole_likelihood_important_var = None
    for index, classId in enumerate(likelihood_by_class.keys()):
        if index == 0:
            merged_whole_likelihood_important_var = copy.deepcopy(likelihood_by_class[classId])
        else:
            merged_whole_likelihood_important_var['dist_'+str(classId)] = copy.deepcopy(likelihood_by_class[classId]['dist_'+str(classId)])
    # Reordering the columns
    non_dist_column_names = [column for column in merged_whole_likelihood_important_var.columns.values if 'dist' not in column]
    dist_column_names = [column for column in merged_whole_likelihood_important_var.columns.values if 'dist' in column]
    reordered_column_names = [*non_dist_column_names, *dist_column_names]
    merged_whole_likelihood_important_var = merged_whole_likelihood_important_var[reordered_column_names]
    
    return merged_whole_likelihood_important_var

def calculate_likelihood_with_important_vars_saving_memory(distribution_folder_path, actLevels, sorted_important_var_by_class,
                                             layerId, distrib_prefix='distribution_important_var_class_'):
    """
    This function executes the experiment to have the likelihood distances
    with only the important variables for different classes. (saving memory verison)
    
    distribution_folder_path: The path of the folder that contains different distributions of each class.
    actLevels: The original activation levels
    sorted_important_var_by_class: The sorted important variables at the desired layer
    layerId: The desired layer to be evaluated
    distrib_prefix: The prefix of the distribution files
    """
    # Generate the likelihood of each class with only the important variables     
    likelihood_by_class = {}
    for classId in sorted_important_var_by_class:
        ## Generate the activation levels for this class
        # Get the important vars
        current_class_important_var_indices = sorted_important_var_by_class[classId]
        # Distorted image activation levels     
        actLevels_of_important_vars = copy.deepcopy(actLevels)
        actLevels_of_important_vars['actLevel'][layerId] = actLevels_of_important_vars['actLevel'][layerId][:, current_class_important_var_indices]
        ## Get the distribution file and evaluate the likelihood        
        current_class_distrib_file_path = path_join(distribution_folder_path, distrib_prefix+str(classId)+'.csv')
        likelihood_by_class[classId] = layer_one_class_likelihood_experim(actLevels_of_important_vars, layerId, 
                                                                                      current_class_distrib_file_path, classId,
                                                                                      use_absolute_module_path=True)
    # Merge the likelihood from different class
    merged_whole_likelihood_important_var = None
    for index, classId in enumerate(likelihood_by_class.keys()):
        if index == 0:
            merged_whole_likelihood_important_var = copy.deepcopy(likelihood_by_class[classId])
        else:
            merged_whole_likelihood_important_var['dist_'+str(classId)] = copy.deepcopy(likelihood_by_class[classId]['dist_'+str(classId)])
    # Reordering the columns
    non_dist_column_names = [column for column in merged_whole_likelihood_important_var.columns.values if 'dist' not in column]
    dist_column_names = [column for column in merged_whole_likelihood_important_var.columns.values if 'dist' in column]
    reordered_column_names = [*non_dist_column_names, *dist_column_names]
    merged_whole_likelihood_important_var = merged_whole_likelihood_important_var[reordered_column_names]
    
    return merged_whole_likelihood_important_var

"""
The following function is just a programmed function to perform a complete experiment, it could be never used in other places.
"""
def complete_experim_with_selected_vars(experim_path, important_variables_per_class, train_distribution, set_actLevels, layerId):
    """
    This function executes a complete whole likelihood calculation experiment according to the provided variables (i.e., neruons)
    
    experim_path: The path for the folder in which the temporary distribution will be stored
    important_variables_per_class: The selected important variables per class
    train_distribution: The training set distribution
    set_actLevels: The original activation levels of the desired dataset
    layerId: The desired layer for the likelihood calculation
    """
    # Build the sorted total important variable(neuron) indices
    sorted_important_var_by_class = {}
    for classId in important_variables_per_class:
        sorted_important_var_by_class[classId] = sorted(list(important_variables_per_class[classId]))
    # Build the mapping dictionary to modify neuron indices
    important_var_map_dict_by_class = {}
    for classId in sorted_important_var_by_class:
        important_var_map_dict_by_class[classId] = build_map_to_index_dict(sorted_important_var_by_class[classId])
        
    ## Take the training set distribution only for the important neurons
    # Take the distribution for different classes
    train_important_var_distrib_by_class = {}
    for classId in sorted_important_var_by_class:
        # Get the important variables of the current class
        current_important_neuron_indices = sorted_important_var_by_class[classId]
        # Take only the desired layer distribution (We calculate the likelihood only based on this)
        train_whole_layer_distribution = train_distribution[train_distribution['layerId'] == layerId].copy(deep=True)
        # Filter the distribution on this layer
        important_var_train_whole_layer_distribution = train_whole_layer_distribution[train_whole_layer_distribution['nodeId'].isin(current_important_neuron_indices)]
        # Map the node Id
        train_important_var_distrib_by_class[classId] = important_var_train_whole_layer_distribution.replace({"nodeId": important_var_map_dict_by_class[classId]}).reset_index(drop=True)
    
    # Save the temporarily generated distribution files by class
    for classId in train_important_var_distrib_by_class:
        save_df_to_csv(path_join(experim_path, 'distribution_important_var_class_'+str(classId)+csv_file_extension), train_important_var_distrib_by_class[classId], sep=' ')

    # Take the activation levels of the desired neurons
    built_set_actLevels_by_class = build_actLevel_important_vars(set_actLevels, sorted_important_var_by_class, layerId)
    
    # Calculate the likelihood with the important variables     
    set_whole_likelihood_important_vars = calculate_likelihood_with_important_vars(experim_path, built_set_actLevels_by_class,
                                             layerId)
    
    # Erase the temporary distribution files
    for classId in train_important_var_distrib_by_class:
        erase_one_file(path_join(experim_path, 'distribution_important_var_class_'+str(classId)+csv_file_extension))
    
    return set_whole_likelihood_important_vars

"""
Normalized version (whcih means the whole likelihoods are normalized by the corresponding standard deviation)
"""

def normalize_whole_distances(layer_whole_distances, class_dist_infos):
    """
    This function normalizes the obtained whole likelihoods by the distance info per class
    """
    for classId in class_dist_infos:
        current_dist_column = dist_column + '_' + str(classId)
        current_class_std = class_dist_infos[classId][std_name]
        layer_whole_distances[current_dist_column] = layer_whole_distances[current_dist_column] / current_class_std
    return layer_whole_distances

def filter_decision_based_on_train_infos_norm_ver(layer_whole_distances, class_dist_infos, std_threshold_coeff=1, original_class=False):
    """
    This function finds the index of the odd decisions based on the likelihood distance (normalized version)

    original_class: Boolean parameter which determines if we are filetring the entries based on the original class(Case: True) or the predicted class(Case: False)
    
    Note: This function supposes that the provided whole likelihood distances are already normalized by the std in "class_dist_infos".
    """
    filtered_index = []
    for index, row in layer_whole_distances.iterrows():
        current_ref_class = None
        if original_class:
            current_ref_class = row[class_column]
        else:
            current_ref_class = row[predicted_class_column]
        current_dist_column = dist_column + '_' + str(int(current_ref_class))
        current_entry_dist = row[current_dist_column]
        current_class_mean = class_dist_infos[str(int(current_ref_class))][mean_name]
        current_class_std = class_dist_infos[str(int(current_ref_class))][std_name]
        if current_entry_dist > current_class_mean / current_class_std + std_threshold_coeff:
            filtered_index.append(index)
    return filtered_index

"""
The following functions are the resources saving version of likelihood evalution. We evaluate only based on the predicted class
when detecting OOD cases, and based on the original class for obtaining the separation thresholds.
"""
def layer_ref_likelihood_experim(registered_actLevel, layerId, hist_prob_path, predicted_class=False, use_absolute_module_path=False, display=True):
    """
    This function executes the experiment of likelihood evaluation according to the predicted class or original class.

    predicted_class: Boolean indicating if we would like to evaluate the likelihoods according to the predicted classes.
    display: Boolean indicating if we would like to display the execution process

    Note: You can use the “use_absolute_module_path” paramter to solve the question of relative path to the desired executable.
    However, you should always provid the "options.dat" file in your current working directory. (an example of this file is provided in the module) 
    """
    ## Experiment preparation
    # Take the corresponding data
    layer_actLevels = registered_actLevel['actLevel'][layerId]
    ground_truths = registered_actLevel['class'].reshape(-1)
    predicted_classes = registered_actLevel['predict_class'].reshape(-1)
    # Determine the referenced class
    ref_class = 'class'
    if predicted_class:
        ref_class = 'predict_class'
    # Create the mapping dictionary
    entry_map_dict = {}
    ref_class_values = registered_actLevel[ref_class].reshape(-1)
    nb_examples = ref_class_values.shape[0]
    for index, class_value in enumerate(ref_class_values):
        if class_value not in entry_map_dict:
            entry_map_dict[class_value] = []
        entry_map_dict[class_value].append(index)
    # Evaluate the likelihood for each class
    eval_class_list = list(entry_map_dict.keys())
    class_list_iterator = tqdm(eval_class_list, desc='Processed class for the likelihood calculation: ') if display else eval_class_list
    layer_set_classes_likelihood = {} # The dictionary that stores temprorary the likelihood of each class
    for entry_class in class_list_iterator:
        # Take the corresponding entries
        current_class_actLevels = layer_actLevels[entry_map_dict[entry_class]]
        # Create the temp folder
        create_directory(temp_path, display=False)
        # Create the folder to store the converted data
        layer_temp_path = path_join(temp_path, str(layerId))
        create_directory(layer_temp_path, display=False)
        # Convert the data
        converted_files = activation_levels_to_required_data_files(layerId, current_class_actLevels, layer_temp_path, display=False)
        # Evaluate likelihood
        likelihood_result = all_files_likelihood_calc(converted_files, layer_temp_path, hist_prob_path, entry_class,
                                                      layerId, with_predict=True, pool_ver=True, nb_divide_task=4,
                                                      use_absolute_module_path=use_absolute_module_path)
        # Create a temporary dataframe stores the likelihood results for the current class
        class_likelihood_df = pd.DataFrame(likelihood_result, columns=distance_with_predict_class_file_headers[:3])
        layer_set_classes_likelihood[entry_class] = class_likelihood_df
        # Erase the content in the temp folder
        erase_files_from_folder(temp_path)
    ## Build the total likelihood dataframe
    # Intialize the result dataframe     
    final_distance_headers = copy.deepcopy(distance_with_predict_class_file_headers)
    layer_distances = pd.DataFrame(columns=final_distance_headers)
    # Build the total distance values
    total_dist_values = np.zeros(nb_examples)
    for entry_class in layer_set_classes_likelihood:
        current_class_likelihood_df = layer_set_classes_likelihood[entry_class]
        current_class_dist_values = current_class_likelihood_df[dist_column].to_numpy()
        total_dist_values[entry_map_dict[entry_class]] = current_class_dist_values
    # Add the class related information
    layer_distances[final_distance_headers[1]] = list(range(nb_examples))
    layer_distances[dist_column] = total_dist_values
    layer_distances[class_column] = ground_truths
    layer_distances[predicted_class_column] = predicted_classes
    layer_distances[final_distance_headers[0]] = layerId # It should be the last because we need entries to correctly assign scalar values
    
    return layer_distances, entry_map_dict

def build_layer_train_set_infos_ref_ver(train_layer_distances, class_list):
    """
    This function builds the statistical information about the distances of the training set inputs.
    
    Note: This function should be applied to a dataframe that contains only the distance referred to the corresponding class of each entry,
    the predicted class or the original class. In other words, it should have only the column named "dist".
    """
    ## Calculate the information and save it
    # Information calculation
    class_dist_infos = {}
    for entry_class in class_list:
        # Intialization
        class_dist_infos[str(entry_class)] = {}
        # Calculation     
        current_class_entries =  train_layer_distances[train_layer_distances[class_column] == entry_class]
        current_class_dists = current_class_entries[dist_column].to_numpy()
        current_class_dist_mean,current_class_dist_std  = get_mean_std(current_class_dists)
        # Register the information
        class_dist_infos[str(int(entry_class))][mean_name] = current_class_dist_mean
        class_dist_infos[str(int(entry_class))][std_name] = current_class_dist_std
    return class_dist_infos

def normalize_distances_ref_ver(layer_distances, map_dict, class_dist_infos):
    """
    This function normalizes the obtained likelihoods by the distance info per class
    """
    # Normalize the distances     
    normalized_distances = np.zeros(layer_distances.shape[0])
    unormalized_distances = layer_distances[dist_column].to_numpy()
    for classId in map_dict:
        current_class_std = class_dist_infos[str(classId)][std_name]
        current_class_entry_distances = unormalized_distances[map_dict[int(classId)]]
        normalized_distances[map_dict[int(classId)]] = current_class_entry_distances / current_class_std
    # Assign the normalized distances     
    layer_distances[dist_column] = normalized_distances
    return layer_distances

def filter_decision_based_on_train_infos_norm_and_ref_ver(layer_distances, class_dist_infos, std_threshold_coeff=1, original_class=False):
    """
    This function finds the index of the odd decisions based on the likelihood distance (normalized version)

    original_class: Boolean parameter which determines if we are filetring the entries based on the original class(Case: True) or the predicted class(Case: False)
    
    Note: This function supposes that the provided whole likelihood distances are already normalized by the std in "class_dist_infos".
    And it is the "referred version", that means we use directly the distances in the "dist" column along for direct filtering.
    """
    filtered_index = []
    for index, row in layer_distances.iterrows():
        current_ref_class = None
        if original_class:
            current_ref_class = row[class_column]
        else:
            current_ref_class = row[predicted_class_column]
        current_entry_dist = row[dist_column]
        current_class_mean = class_dist_infos[str(int(current_ref_class))][mean_name]
        current_class_std = class_dist_infos[str(int(current_ref_class))][std_name]
        if current_entry_dist > current_class_mean / current_class_std + std_threshold_coeff:
            filtered_index.append(index)
    return filtered_index

def calculate_likelihood_with_important_vars_saving_memory_ref_ver(distribution_folder_path, registered_actLevel, sorted_important_var_by_class,
                                             layerId, predicted_class=False, use_absolute_module_path=False, display=True, distrib_prefix='distribution_important_var_class_'):
    """
    This function executes the experiment to have the likelihood distances
    with only the important variables for different classes. (saving memory and referred verison)
    
    distribution_folder_path: The path of the folder that contains different distributions of each class.
    actLevels: The original activation levels
    sorted_important_var_by_class: The sorted important variables at the desired layer
    layerId: The desired layer to be evaluated
    predicted_class: Boolean indicating if we would like to evaluate the likelihoods according to the predicted classes.
    use_absolute_module_path: Boolean indicating if we use the absolute path of the likelihood program file.
    display: Boolean indicating if we display the progress bar.
    distrib_prefix: The prefix of the distribution files.
    """
    ## Experiment preparation
    # Take the corresponding data
    layer_actLevels = registered_actLevel['actLevel'][layerId]
    ground_truths = registered_actLevel['class'].reshape(-1)
    predicted_classes = registered_actLevel['predict_class'].reshape(-1)
    # Determine the referenced class
    ref_class = 'class'
    if predicted_class:
        ref_class = 'predict_class'
    # Create the mapping dictionary
    entry_map_dict = {}
    ref_class_values = registered_actLevel[ref_class].reshape(-1)
    nb_examples = ref_class_values.shape[0]
    for index, class_value in enumerate(ref_class_values):
        if class_value not in entry_map_dict:
            entry_map_dict[class_value] = []
        entry_map_dict[class_value].append(index)
    # Evaluate the likelihood for each class
    eval_class_list = list(entry_map_dict.keys())
    class_list_iterator = tqdm(eval_class_list, desc='Processed class for the likelihood calculation: ') if display else eval_class_list
    layer_set_classes_likelihood = {} # The dictionary that stores temprorary the likelihood of each class
    for entry_class in class_list_iterator:
        # Get the important vars
        current_class_important_var_indices = sorted_important_var_by_class[entry_class]
        # Take the corresponding entries
        current_class_actLevels = copy.deepcopy(layer_actLevels[entry_map_dict[entry_class]])
        # Take only the activation levels on the significant neurons
        current_class_actLevels_important_vars = current_class_actLevels[:, current_class_important_var_indices]
        # Create the temp folder
        create_directory(temp_path, display=False)
        # Create the folder to store the converted data
        layer_temp_path = path_join(temp_path, str(layerId))
        create_directory(layer_temp_path, display=False)
        # Convert the data
        converted_files = activation_levels_to_required_data_files(layerId, current_class_actLevels_important_vars,
                                                                   layer_temp_path, display=False)
        ## Get the distribution file and evaluate the likelihood        
        current_class_distrib_file_path = path_join(distribution_folder_path, distrib_prefix+str(entry_class)+'.csv')
        likelihood_result = all_files_likelihood_calc(converted_files, layer_temp_path, current_class_distrib_file_path, entry_class,
                                                      layerId, with_predict=True, pool_ver=True, nb_divide_task=4,
                                                      use_absolute_module_path=use_absolute_module_path)
        # Create a temporary dataframe stores the likelihood results for the current class
        class_likelihood_df = pd.DataFrame(likelihood_result, columns=distance_with_predict_class_file_headers[:3])
        layer_set_classes_likelihood[entry_class] = class_likelihood_df
        # Erase the content in the temp folder
        erase_files_from_folder(temp_path)
    ## Build the total likelihood dataframe
    # Intialize the result dataframe     
    final_distance_headers = copy.deepcopy(distance_with_predict_class_file_headers)
    layer_distances = pd.DataFrame(columns=final_distance_headers)
    # Build the total distance values
    total_dist_values = np.zeros(nb_examples)
    for entry_class in layer_set_classes_likelihood:
        current_class_likelihood_df = layer_set_classes_likelihood[entry_class]
        current_class_dist_values = current_class_likelihood_df[dist_column].to_numpy()
        total_dist_values[entry_map_dict[entry_class]] = current_class_dist_values
    # Add the class related information
    layer_distances[final_distance_headers[1]] = list(range(nb_examples))
    layer_distances[dist_column] = total_dist_values
    layer_distances[class_column] = ground_truths
    layer_distances[predicted_class_column] = predicted_classes
    layer_distances[final_distance_headers[0]] = layerId # It should be the last because we need entries to correctly assign scalar values
    
    return layer_distances, entry_map_dict