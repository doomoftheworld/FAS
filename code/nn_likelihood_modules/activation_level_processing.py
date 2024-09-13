from common_imports import copy

"""
This file contains the processing of the generated activation levels
"""
def build_correct_actLevels(actLevels):
    """
    This function builds the activation levels only on the correctly examples.
    
    actLevels: The original actLevels.
    """
    correct_actLevels = copy.deepcopy(actLevels)
    correct_example_bools = (correct_actLevels['class'] == correct_actLevels['predict_class']).reshape(-1)
    for info_key in correct_actLevels:
        if info_key == 'actLevel':
            for layerId in correct_actLevels[info_key]:
                correct_actLevels[info_key][layerId] = correct_actLevels[info_key][layerId][correct_example_bools]
        else:
            correct_actLevels[info_key] = correct_actLevels[info_key][correct_example_bools]
    
    return correct_actLevels