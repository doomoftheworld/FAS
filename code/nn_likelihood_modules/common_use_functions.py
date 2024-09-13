"""
Module with the defined common use functions (normally file system operation functions and some pratical use functions)
"""
import os
import csv
import json
import shutil
import math
import random
from datetime import datetime
from common_imports import *

"""
Functions of file systems operations
"""
# Activation levels conservation function
def write_csv(data, path):
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerows(data)
        
        f.close()

def contents_of_folder(folder_path):
    return os.listdir(folder_path)

def create_directory(dir, display=True):
    """
    display: Boolean indicating the folder creation information ("success" or "already exists")
    """
    try:
        # Create target Directory
        os.mkdir(dir)
        if display:
            print("Directory " , dir ,  " Created ") 
    except FileExistsError:
        if display:
            print("Directory " , dir ,  " already exists")

def erase_files_from_folder(folder_path):
    shutil.rmtree(folder_path)

def erase_one_file(file_path):
    # If file exists, delete it.
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        # If it fails, inform the user.
        print("Error: %s file not found" % file_path)

def read_content_path(path):
    contents = {}
    for file in os.listdir(path):
        contents[file] = os.path.join(path, file)
    return contents

def path_join(*path_parts):
    return os.path.join(*path_parts)

def store_list_as_json(path, list_to_save):
    with open(path, 'w') as fp:
        json.dump(list_to_save, fp)
        fp.close()

def store_dict_as_json(path, dict_to_save):
    with open(path, 'w') as fp:
        json.dump(dict_to_save, fp)
        fp.close()

def load_json(fp):
    return json.load(fp)

def load_json_by_path(path):
    loaded_json = None
    with open(path) as fp:
        loaded_json = json.load(fp)
        fp.close()
    return loaded_json

def read_csv_to_pd_df(csv_path, first_line_as_head=True, display=False):
    df = None
    if first_line_as_head:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)

    if display:
        print(df.to_string())
    return df

def save_list_to_csv(csv_path, list, index=None, headers=None, save_index=False, save_headers=True, sep=','):
    if index is not None:
        if headers is not None:
            pd.DataFrame(np.array(list), columns=headers, index=index).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
        else:
            pd.DataFrame(np.array(list), index=index).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
    else:
        if headers is not None:
            pd.DataFrame(np.array(list), columns=headers).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
        else:
            pd.DataFrame(np.array(list)).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')

def save_list_to_csv_without_np_conversion(csv_path, list, index=None, headers=None, save_index=False, save_headers=True, sep=','):
    if index is not None:
        if headers is not None:
            pd.DataFrame(list, columns=headers, index=index).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
        else:
            pd.DataFrame(list, index=index).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
    else:
        if headers is not None:
            pd.DataFrame(list, columns=headers).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
        else:
            pd.DataFrame(list).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')

def save_df_to_csv(csv_path, df, save_index=False, save_headers=True, sep=','):
    df.to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+')

def save_content_to_file(data, file_path, with_newline=True):
    # Data should be a list of string
    with open(file_path, 'w+', encoding='UTF-8') as f:  
        if with_newline:
            f.write('\n'.join(data) + '\n')
        else:
            f.writelines(data)

def download_image_from_url(img_url, save_path=None):
    """
    The save_path should contains the filename of the image
    """
    img = Image.open(requests.get(img_url, stream = True).raw)
    if save_path is not None:
        img.save(save_path)
    return img

"""
Pratical use functions
"""
def get_actual_time():
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string

def merge_dicts(*dicts):
    merged_dict = dicts[0].copy()
    nb_dicts = len(dicts)
    if nb_dicts > 1:
        for index in range(1, nb_dicts):
            merged_dict.update(dicts[index])
    return merged_dict

def join_string(str_list, delimiter='_'):
    return delimiter.join(str_list)

def str_first_part_split_from_r(str, delimiter='.'):
    return str.rsplit(delimiter,1)[0]

def str_second_part_split_from_r(str, delimiter='.'):
    return str.rsplit(delimiter,1)[1]

def str_first_part_split_from_l(str, delimiter='_'):
    return str.split(delimiter, 1)[0]

def str_second_part_split_from_l(str, delimiter='_'):
    return str.split(delimiter, 1)[1]

def generate_loader(X, Y, batch_size=100):
    torch_inputs = torch.from_numpy(X)
    torch_labels = torch.from_numpy(Y)

    # Create the dataset
    torch_data_set = TensorDataset(torch_inputs, torch_labels)

    # Data Loader
    return DataLoader(torch_data_set, batch_size=batch_size) 

def get_swap_dict(d):
    """
    Function to be used to swap the keys and the values in a dictionary to create a swapped dictionary
    """
    return {v: k for k, v in d.items()}

def torch_dataset_to_numpy(torch_dataset):
    """
    This function get the data from a torch dataset to numpy array

    torch_dataset: A pytorch dataset object
    """
    no_divide_into_batch_loader = DataLoader(torch_dataset, batch_size=len(torch_dataset), shuffle=False, num_workers=0)
    dataset_X = next(iter(no_divide_into_batch_loader))[0].numpy()
    dataset_y = next(iter(no_divide_into_batch_loader))[1].numpy()
    return dataset_X, dataset_y

def most_common(one_list):
    """
    This function get the most common element from a list
    """
    data = Counter(one_list)
    return max(one_list, key=data.get)

def build_map_to_index_dict(one_list):
    """
    This function build a dictionary that maps list values to their index
    """
    return {k: v for v, k in enumerate(one_list)}


def floor_int_conversion(one_float):
    """
    This function applies the floor conversion from float to int.
    """
    return math.floor(one_float)

def random_sample_from_list(one_list, nb_samples):
    """
    This function applies a random sampling from the provided list
    """
    return random.sample(one_list, nb_samples)

"""
Simple activation functions
"""
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Takes the input and applies the softmax function to it
    
    Args:
        x: Numpy array of shape []
    
    Returns:
        np.ndarray: the softmax-ed input
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1).reshape(-1,1)