from common_imports import *
from defined_network_structure import *

"""
This module includes functions to apply predictions with pytorch models and essential function to get model information.
"""

def load_model(pt_filepath):
    """
    This function load the pytorch model from the provided .pt filepath (directly whole model version, not with state dict)
    """
    return torch.load(pt_filepath)

def load_state_dict(model, state_dict_filepath):
    """
    This function load the state dict to the given pytorch model
    """
    model.load_state_dict(torch.load(state_dict_filepath))

def single_example_feature_to_tensor(X_features, list_form=True):
    """
    This function convert a list form features to pytorch tensor

    list_form: boolean indicates if the provided data is in list form (True) or array form (False)
    """
    X_array = None
    if list_form:
        X_array = np.array(X_features).reshape(1,-1)
    else:
        X_array = X_features.reshape(1,-1)
    return torch.from_numpy(X_array)

def model_predict_single_data_cpu(model, X_single, display=False):
    """
    This function applies the prediciton of a single data based on the provided pytorch model
    """
    # Move the model and the data to cpu
    model = model.cpu()
    X_single = X_single.float().cpu()
    # Get the output (it is log probability because of the defined network structure (most of the structure are with LogSoftmax))
    log_proba = model(X_single).detach().cpu().numpy()
    # Get the predicted class and probability
    proba = np.exp(log_proba.reshape(-1))
    predicted_class = np.argmax(proba)
    class_orders = list(range(len(proba)))
    class_proba_infos = dict(zip(class_orders, proba.tolist()))
    confidence_score = class_proba_infos[predicted_class]
    # Display
    if display:
        print('The input single data is predicted as class', str(predicted_class), 'with the confidence score of', str(confidence_score), '.')
    return predicted_class, confidence_score, class_proba_infos


"""
Function to get model information
"""
def get_model_parameters(model, to_numpy=False):
    """
    This function returns all the parameters (with their values) from the provided model.

    model: the provided pytorch model
    to_numpy: Parameter that controls if we return directly the parameter values as numpy array
    """
    # The final result
    param_dict = {}
    # Apply the parameter dictionary building process
    for name, param in model.named_parameters():
        # Get the name components, generally we have the layer name and then the parameter type (weights or bias)
        name_components = name.rsplit(".", 1)
        # Get the different parts of name
        layer_name = name_components[0]
        param_type = name_components[1]
        # Store the parameter in the final result dictionary
        if layer_name not in param_dict:
            param_dict[layer_name] = {}
        if to_numpy:
            param_dict[layer_name][param_type] = copy.deepcopy(param.data.detach().cpu().numpy())
        else:
            param_dict[layer_name][param_type] = param.data
    
    return param_dict

