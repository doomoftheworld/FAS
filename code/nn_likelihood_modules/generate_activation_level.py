"""
Functions that generate the activation levels for the fullyconnected network (hard coded version)
"""

from common_imports import *
from experim_neural_network import evaluate_gpu_register_ver, evaluate_gpu_register_ver_with_predicts, evaluate_cpu_register_ver_with_predicts

"""
Activation levels generation
"""
def obtain_activation_levels(model, data_loader, set_name, with_predict_class=True, loss_type='nll'):
    actLevel = []
    model.activate_registration()
    if torch.cuda.is_available():
        if with_predict_class:
            _,_,actLevel = evaluate_gpu_register_ver_with_predicts(model.cuda(), data_loader, set_name, loss_type=loss_type)
        else:
            _,_,actLevel = evaluate_gpu_register_ver(model.cuda(), data_loader, set_name, loss_type=loss_type)
    model.deactivate_registration()
    return actLevel

def obtain_activation_levels_with_predict_cpu_ver(model, data_loader, set_name, loss_type='nll'):
    actLevel = []
    model.activate_registration()
    _,_,actLevel = evaluate_cpu_register_ver_with_predicts(model.cpu(), data_loader, set_name, loss_type=loss_type)
    model.deactivate_registration()
    return actLevel