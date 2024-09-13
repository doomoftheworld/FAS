"""
General constant variables
"""
np_file_extension = 'npy' # Potentially could be changed to '.npy'
csv_file_extension = '.csv'
model_state_file_keyword = 'model'
model_config_file_keyword = 'configuration'
"""
Likelihood execution variables
"""
temp_path = '.\\temp\\'
single_class_likelihood_path = 'distr\\histoSigNodeSingleClassLikelihood.py'
python_command = 'python'
data_list_name = 'dataFileList'
dat_extension = '.dat'
distance_with_predict_class_file_headers = ['layerId', 'entryId', 'dist', 'classId', 'predicted_classId']
whole_distance_headers = ['layerId', 'entryId'] # To be build
whole_distance_headers_density_ver = ['layerId', 'entryId', 'classId', 'predicted_classId'] # To be build
dist_column = 'dist'
class_column = 'classId'
predicted_class_column = 'predicted_classId'
mean_name = 'mean'
std_name = 'std'
torch_ext = '.pt'
small_constant = 0.0000000000000001
single_norm_CPD_path = 'single_norm_CPD.py'
"""
Cost analysis variables
"""
# The binary class list
binary_class_list = [0, 1]
# Must configured parameter when using the likelihood modules
nn_likelihood_module_absolute_path = '..\\nn_likelihood_modules'