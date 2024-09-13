"""
This file contains the functions for the deep k-nearst neighbors OMS detection.
"""
import faiss
from common_imports import np, copy, tqdm

"""
Functions for the original KNN algorithm
"""
def normalize_feature_vecs_knn(actLevels, last_hidden_layerId):
    """
    This function normalizes the vector from the last hidden layer activation levels (i.e., unnormalized feature vectors)

    actLevels: The extracted activation levels.
    last_hidden_layerId: The last hidden layer Id.
    """
    feature_vecs = copy.deepcopy(actLevels['actLevel'][last_hidden_layerId])
    feature_vec_norms = np.linalg.norm(feature_vecs, axis=1, keepdims=True)
    zs = feature_vecs / feature_vec_norms

    return zs

# Deep k-nearst neighbors OOD detection function
def faiss_knn_search(search_index, feature_vecs, k, display=False):
    """
    Apply the faiss knn similarity search by iteration.
    
    search_index: The faiss search index.
    feature_vecs: The vectors to apply the search.
    k: The number of nearst neighbors.
    display: The boolean that indicates if we want to display the progress bar.
    """
    # Intialize the evaluated distances and indices        
    D = []
    I = []
    # Get the number of examples
    nb_examples = feature_vecs.shape[0]
    # Build the iterate index list
    index_progress_bar = None
    if display:
        index_progress_bar = tqdm(list(range(nb_examples)), desc='Processed examples')
    else:
        index_progress_bar = list(range(nb_examples))
    # Iterate over the examples
    for index in index_progress_bar:
        # Get the current vector
        current_vec = feature_vecs[index].reshape(1,-1)
        # Apply the search
        current_D, current_I = search_index.search(current_vec, k)
        # Add the results
        D.append(current_D)
        I.append(current_I)
    # Stack the results
    D = np.vstack(D)
    I = np.vstack(I)
    
    return D, I

def k_nearst_neighbor_scores(train_index, test_zs, k):
    """
    This function obtains the k-nearst neighbor scores.

    train_index: The "faiss" index for similarity search.
    test_zs: The normalized test set feature vectors that contain potentially OOD examples.
    k: The number of considered nearst neighbors.
    """
    # D represents distances, I represents index
    D, I = faiss_knn_search(train_index, test_zs, k, display=True)
    # Evaluate the scores (S)
    S = -D[:,-1]

    return S

def k_nearst_neighbor_OOD_detection(test_scores, threshold):
    """
    This function uses the k-nearst neighbor scores used for the OOD detection.

    test_scores: The k-nearst neighbor scores of the test set with potential OOD examples.
    threshold: The OOD detection threshold.
    """
    # OOD detection with the given threshold
    ood_result = np.less(test_scores, threshold).astype(int)

    return ood_result

def k_nearst_neighbor_OOD_detection_pipeline(train_index, test_zs, k, threshold):
    """
    This function executes the complete k-nearst neighbor OOD detection pipeline.

    train_index: The "faiss" index for similarity search.
    test_zs: The normalized test set feature vectors that contain potentially OOD examples.
    k: The number of considered nearst neighbors.
    threshold: The OOD detection threshold.
    """
    # D represents distances, I represents index
    D, I = faiss_knn_search(train_index, test_zs, k, display=True)
    # Evaluate the scores (S)
    S = -D[:,-1]
    # OOD detection with the given threshold
    ood_result = np.less(S, threshold).astype(int)

    return ood_result

def experim_ood_detection_knn(train_index, test_zs, test_actLevels, k, threshold, set_name):
    """
    This function executes the complete k-nearst neighbor OOD detection experiment.

    train_index: The "faiss" index for similarity search.
    test_zs: The normalized test set feature vectors that contain potentially OOD examples.
    test_actLevels: the activation level information that contains also the original and predicted class.
    k: The number of considered nearst neighbors.
    threshold: The OOD detection threshold.
    set_name: The OOD set name.
    """
    # Get the ground truths and predicted class
    groundtruths = test_actLevels['class']
    predictions = test_actLevels['predict_class']
    nb_examples = groundtruths.shape[0]
    # Execute the experiment
    ood_result = k_nearst_neighbor_OOD_detection_pipeline(train_index, test_zs, k, threshold)
    ood_result_bool = ood_result.astype(bool)
    # Number of InD and OOD examples and their percentages
    nb_ood = ood_result.sum()
    nb_ind = ood_result.shape[0] - nb_ood
    ood_percent = nb_ood / ood_result.shape[0]
    ind_percent = nb_ind / ood_result.shape[0]
    # Get the total, ind and ood accuracy
    acc_total = np.sum(groundtruths == predictions) / nb_examples
    acc_ood = np.sum(groundtruths[ood_result_bool] == predictions[ood_result_bool]) / nb_ood
    acc_ind = np.sum(groundtruths[np.invert(ood_result_bool)] == predictions[np.invert(ood_result_bool)]) / nb_ind
    # Display the results
    print('The number of OOD examples in', set_name, 'set:', nb_ood)
    print('The number of InD examples in', set_name, 'set:', nb_ind)
    print('The percentage of OOD examples in', set_name, 'set:', ood_percent)
    print('The percentage of InD examples in', set_name, 'set:', ind_percent)
    print('The total accuracy in', set_name, 'set:', acc_total)
    print('The accuracy on the OOD examples in', set_name, 'set:', acc_ood)
    print('The accuracy on the InD examples in', set_name, 'set:', acc_ind)

    return [set_name, nb_ood, nb_ind, ood_percent, ind_percent, acc_total, acc_ood, acc_ind]

# """
# Functions for the modified KNN algorithm (by class version)

# Note: This by class version applies the knn search from one normalized feature vector to the ones of its predicted class in the training set
# """
# def build_by_class_normalized_zs(actLevels, last_hidden_layerId, predict_class_ver=False):
#     """
#     This function builds the by-class normalized vectors from the last hidden layer activation levels (i.e., unnormalized feature vectors)

#     actLevels: The extracted activation levels.
#     last_hidden_layerId: The last hidden layer Id.
#     predict_class_ver: Build the class index dictionary with the predicted class or the original class
#     """
#     # Normalize the feature vectors
#     feature_vecs = actLevels['actLevel'][last_hidden_layerId]
#     feature_vec_norms = np.linalg.norm(feature_vecs, axis=1, keepdims=True)
#     zs = feature_vecs / feature_vec_norms
#     # Determine the referenced class tye
#     ref_class_type = 'class'
#     if predict_class_ver:
#         ref_class_type = 'predict_class'
#     # Get the referred class for all examples
#     ref_classes = actLevels[ref_class_type].reshape(-1)
#     # Separate the feature vectors by their original class
#     class_index_dict = {}
#     uniq_classes = list(np.unique(ref_classes))
#     for uniq_class in uniq_classes:
#         class_index_dict[uniq_class] = []
#     for index, val in enumerate(ref_classes):
#         class_index_dict[val].append(index)
#     # Build the by-class normalized feature vectors
#     zs_by_class = {}
#     for uniq_class in class_index_dict:
#         zs_by_class[uniq_class] = zs[class_index_dict[uniq_class]]

#     return zs, zs_by_class, class_index_dict

# # Deep k-nearst neighbors OOD detection function (by class version)
# def k_nearst_neighbor_scores_by_class(train_indices, test_zs, test_class_index_dict, k):
#     """
#     This function obtains the k-nearst neighbor scores (by class version).

#     train_indices: The "faiss" index of all classes for similarity search.
#     test_zs: The normalized test set feature vectors that contain potentially OOD examples.
#     test_class_index_dict: The dictionary indicating the positions of entries from different classes.
#     k: The number of considered nearst neighbors.
#     """
#     # Intialize the results
#     S_total = np.zeros(test_zs.shape[0])
#     S_by_class = {}
#     # Evaluate the scores
#     for uniq_class in test_class_index_dict:
#         # D represents distances, I represents index
#         class_D, class_I = train_indices[uniq_class].search(test_zs[test_class_index_dict[uniq_class]], k)
#         # Evaluate the scores (S)
#         class_S = -class_D[:,-1]
#         # Assign the scores of the current class
#         S_by_class[uniq_class] = class_S
#         S_total[test_class_index_dict[uniq_class]] = class_S

#     return S_by_class, S_total

# def k_nearst_neighbor_OOD_detection_by_class(test_scores_total, test_class_index_dict, thresholds):
#     """
#     This function uses the k-nearst neighbor scores used for the OOD detection (by class version).

#     test_scores_total: The k-nearst neighbor scores of the test set with potential OOD examples.
#     test_class_index_dict: The dictionary indicating the positions of entries from different classes.
#     thresholds: The OOD detection thresholds for different classes.
#     """
#     # OOD detection with the given thresholds
#     ood_result_by_class = {}
#     ood_result_total = np.zeros(test_scores_total.shape[0])
#     for uniq_class in test_class_index_dict:
#         class_ood_result = np.less(test_scores_total[test_class_index_dict[uniq_class]], thresholds[uniq_class]).astype(int)
#         ood_result_by_class[uniq_class] = class_ood_result
#         ood_result_total[test_class_index_dict[uniq_class]] = class_ood_result

#     return ood_result_by_class, ood_result_total

# def k_nearst_neighbor_OOD_detection_by_class_pipeline(train_indices, test_zs, test_class_index_dict, k, thresholds):
#     """
#     This function executes the complete k-nearst neighbor OOD detection pipeline (by class version).

#     train_indices: The "faiss" index of all classes for similarity search.
#     test_zs: The normalized test set feature vectors that contain potentially OOD examples.
#     test_class_index_dict: The dictionary indicating the positions of entries from different classes.
#     k: The number of considered nearst neighbors.
#     thresholds: The OOD detection thresholds for different classes.
#     """
#     # Intialize the results
#     S_total = np.zeros(test_zs.shape[0])
#     # Evaluate the scores
#     for uniq_class in test_class_index_dict:
#         # D represents distances, I represents index
#         class_D, class_I = train_indices[uniq_class].search(test_zs[test_class_index_dict[uniq_class]], k)
#         # Evaluate the scores (S)
#         class_S = -class_D[:,-1]
#         # Assign the scores of the current class
#         S_total[test_class_index_dict[uniq_class]] = class_S

#     # OOD detection with the given thresholds
#     ood_result_by_class = {}
#     ood_result_total = np.zeros(S_total.shape[0])
#     for uniq_class in test_class_index_dict:
#         class_ood_result = np.less(S_total[test_class_index_dict[uniq_class]], thresholds[uniq_class]).astype(int)
#         ood_result_by_class[uniq_class] = class_ood_result
#         ood_result_total[test_class_index_dict[uniq_class]] = class_ood_result

#     return ood_result_by_class, ood_result_total

# def experim_ood_detection_knn_by_class(train_indices, test_zs_total, test_class_index_dict, k, train_thresholds, set_name):
#     """
#     This function executes the complete k-nearst neighbor OOD detection experiment (by class version).

#     train_indices: The "faiss" index of all classes for similarity search.
#     test_zs: The normalized test set feature vectors that contain potentially OOD examples.
#     test_class_index_dict: The dictionary indicating the positions of entries from different classes.
#     k: The number of considered nearst neighbors.
#     thresholds: The OOD detection thresholds for different classes.
#     set_name: The OOD set name.
#     """
#     # Execute the experiment
#     ood_result_by_class, ood_result_total = k_nearst_neighbor_OOD_detection_by_class_pipeline(train_indices, test_zs_total,
#                                                                                                     test_class_index_dict, k, train_thresholds)
#     nb_detected_ood_total = ood_result_total.sum()
#     print('The detected percentage of OOD examples:', nb_detected_ood_total / ood_result_total.shape[0])

# """
# Functions for the modified KNN algorithm (by class version and using the significant neurons)

# Note: This by class version applies the knn search from one normalized feature vector to the ones of its predicted class in the training set
# """
# def build_by_class_normalized_zs_significant_neurons(actLevels, last_hidden_layerId, significant_neuron_indices, predict_class_ver=False):
#     """
#     This function builds the by-class normalized vectors from the last hidden layer activation levels (i.e., unnormalized feature vectors).
#     It will only consider the evaluated important neurons.

#     actLevels: The extracted activation levels.
#     last_hidden_layerId: The last hidden layer Id.
#     significant_neuron_indices: The significant neurons indices for each class.
#     predict_class_ver: Build the class index dictionary with the predicted class or the original class
#     """
#     # Normalize the feature vectors
#     feature_vecs = actLevels['actLevel'][last_hidden_layerId]
#     feature_vec_norms = np.linalg.norm(feature_vecs, axis=1, keepdims=True)
#     zs = feature_vecs / feature_vec_norms
#     # Determine the referenced class tye
#     ref_class_type = 'class'
#     if predict_class_ver:
#         ref_class_type = 'predict_class'
#     # Get the referred class for all examples
#     ref_classes = actLevels[ref_class_type].reshape(-1)
#     # Separate the feature vectors by their original class
#     class_index_dict = {}
#     uniq_classes = list(np.unique(ref_classes))
#     for uniq_class in uniq_classes:
#         class_index_dict[uniq_class] = []
#     for index, val in enumerate(ref_classes):
#         class_index_dict[val].append(index)
#     # Build the by-class normalized feature vectors
#     zs_by_class = {}
#     for uniq_class in class_index_dict:
#         zs_by_class[uniq_class] = zs[class_index_dict[uniq_class]][:, significant_neuron_indices[uniq_class]]

#     return zs_by_class, class_index_dict

# def k_nearst_neighbor_scores_by_class_sig_ver(train_indices, test_zs_by_class, test_class_index_dict, k):
#     """
#     This function obtains the k-nearst neighbor scores (by class and significant neuron version).

#     train_indices: The "faiss" index of all classes for similarity search.
#     test_zs_by_class: The normalized test set feature vectors that contain potentially OOD examples for each class.
#     test_class_index_dict: The dictionary indicating the positions of entries from different classes.
#     k: The number of considered nearst neighbors.

#     Note: This function is different than the previous one because it builds the scores according to the normalized
#     feature vectors by class not directly the whole feature vectors.
#     """
#     # Determine the number of test examples according to the normalized feature vectors by class
#     nb_examples = np.sum([test_zs_by_class[uniq_class].shape[0] for uniq_class in test_zs_by_class])
#     # Intialize the results
#     S_total = np.zeros(nb_examples)
#     S_by_class = {}
#     # Evaluate the scores
#     for uniq_class in test_class_index_dict:
#         # D represents distances, I represents index
#         class_D, class_I = train_indices[uniq_class].search(test_zs_by_class[uniq_class], k)
#         # Evaluate the scores (S)
#         class_S = -class_D[:,-1]
#         # Assign the scores of the current class
#         S_by_class[uniq_class] = class_S
#         S_total[test_class_index_dict[uniq_class]] = class_S

#     return S_by_class, S_total

# def k_nearst_neighbor_OOD_detection_by_class_pipeline_sig_ver(train_indices, test_zs_by_class, test_class_index_dict, k, thresholds):
#     """
#     This function executes the complete k-nearst neighbor OOD detection pipeline (by class and significant neuron version).

#     train_indices: The "faiss" index of all classes for similarity search.
#     test_zs_by_class: The normalized test set feature vectors that contain potentially OOD examples for each class.
#     test_class_index_dict: The dictionary indicating the positions of entries from different classes.
#     k: The number of considered nearst neighbors.
#     thresholds: The OOD detection thresholds for different classes.

#     Note: This function is different than the previous one because it builds the scores according to the normalized
#     feature vectors by class not directly the whole feature vectors.
#     """
#     # Determine the number of test examples according to the normalized feature vectors by class
#     nb_examples = np.sum([test_zs_by_class[uniq_class].shape[0] for uniq_class in test_zs_by_class])
#     # Intialize the results
#     S_total = np.zeros(nb_examples)
#     # Evaluate the scores
#     for uniq_class in test_class_index_dict:
#         # D represents distances, I represents index
#         class_D, class_I = train_indices[uniq_class].search(test_zs_by_class[uniq_class], k)
#         # Evaluate the scores (S)
#         class_S = -class_D[:,-1]
#         # Assign the scores of the current class
#         S_total[test_class_index_dict[uniq_class]] = class_S

#     # OOD detection with the given thresholds
#     ood_result_by_class = {}
#     ood_result_total = np.zeros(S_total.shape[0])
#     for uniq_class in test_class_index_dict:
#         class_ood_result = np.less(S_total[test_class_index_dict[uniq_class]], thresholds[uniq_class]).astype(int)
#         ood_result_by_class[uniq_class] = class_ood_result
#         ood_result_total[test_class_index_dict[uniq_class]] = class_ood_result

#     return ood_result_by_class, ood_result_total

# def experim_ood_detection_knn_by_class_sig_ver(train_indices, test_zs_by_class, test_class_index_dict, k, train_thresholds, set_name):
#     """
#     This function executes the complete k-nearst neighbor OOD detection experiment (by class and significant neuron version).

#     train_indices: The "faiss" index of all classes for similarity search.
#     test_zs: The normalized test set feature vectors that contain potentially OOD examples.
#     test_class_index_dict: The dictionary indicating the positions of entries from different classes.
#     k: The number of considered nearst neighbors.
#     thresholds: The OOD detection thresholds for different classes.
#     set_name: The OOD set name.
#     """
#     # Execute the experiment
#     ood_result_by_class, ood_result_total = k_nearst_neighbor_OOD_detection_by_class_pipeline_sig_ver(train_indices, test_zs_by_class,
#                                                                                                     test_class_index_dict, k, train_thresholds)
#     nb_detected_ood_total = ood_result_total.sum()
#     print('The detected percentage of OOD examples:', nb_detected_ood_total / ood_result_total.shape[0])

# def accuracy_eval_with_correct_bools(actLevels, ind_correct_bools, ood_correct_bools):
#     """
#     This function executes the evaluation of the accuray results related to the OOD detection with the booleans indicating the InD and OOD examples.

#     actLevels: The extracted activation levels that contais also example-related information (original class, predicted class etc.)
#     ind_correct_bools: The booleans indicating the ind examples.
#     ood_correct_bools: The booleans indicating the ood examples.
#     """
#     test_acc = (actLevels['class'] == actLevels['predict_class']).sum() / actLevels['class'].shape[0]
#     ind_test_acc = (actLevels['class'][ind_correct_bools] 
#                     == actLevels['predict_class'][ind_correct_bools]).sum() / actLevels['class'][ind_correct_bools].shape[0]
#     ood_test_acc = (actLevels['class'][ood_correct_bools] 
#                     == actLevels['predict_class'][ood_correct_bools]).sum() / actLevels['class'][ood_correct_bools].shape[0]
#     print("The accuracy of the test set:", test_acc)
#     print("The InD accuracy of the test set:", ind_test_acc)
#     print("The OOD accuracy of the test set:", ood_test_acc)
    
#     return test_acc, ind_test_acc, ood_test_acc

"""
Functions for the modified KNN algorithm (using the significant neurons but the used significant neurons are different for distinct examples)

Note: This by class version applies the knn search from one normalized feature vector to the ones in the training set (i.e., all the training set examples)
"""
def build_search_index_by_class(zs_by_class):
    """
    This function builds the search index from faiss with the provided normalized feature vectors by class.

    zs_by_class: The normalized feature vectors by class.
    """
    search_indices = {}
    for uniq_class in zs_by_class:
        current_class_index = faiss.IndexFlatL2(zs_by_class[uniq_class].shape[1])
        current_class_index.add(zs_by_class[uniq_class])
        search_indices[uniq_class] = current_class_index

    return search_indices

def build_zs_sig_by_class(zs, actLevels, significant_neuron_indices, predict_class_ver=False):
    """
    This function builds the by-class normalized vectors with only the significant neurons. This version begins directly 
    from the normalized feature vectors (i.e., zs), and the variable "actLevels" is only for mapping dictionary evaluation.
    
    zs: The original normalized feature vectors.
    actLevels: The extracted activation levels.
    last_hidden_layerId: The last hidden layer Id.
    significant_neuron_indices: The significant neurons indices for each class.
    predict_class_ver: Build the class index dictionary with the predicted class or the original class
    """
    # Determine the referenced class tye
    ref_class_type = 'class'
    if predict_class_ver:
        ref_class_type = 'predict_class'
    # Get the referred class for all examples
    ref_classes = actLevels[ref_class_type].reshape(-1)
    # Separate the feature vectors by the referred class
    class_index_dict = {}
    uniq_classes = list(np.unique(ref_classes))
    for uniq_class in uniq_classes:
        class_index_dict[uniq_class] = []
    for index, val in enumerate(ref_classes):
        class_index_dict[val].append(index)
    # Build the by-class normalized feature vectors
    zs_by_class = {}
    for uniq_class in class_index_dict:
        zs_by_class[uniq_class] = zs[class_index_dict[uniq_class]][:, significant_neuron_indices[uniq_class]]

    return zs_by_class, class_index_dict

def knn_scores_sig_ver(train_indices, test_zs_by_class, test_class_index_dict, k):
    """
    This function obtains the k-nearst neighbor scores (by class and significant neuron version).
    This is the same function as the one in the previous by-class version (i.e., which use only the examples from the
    referred class to build the search index and not all examples). We just rename it and build another one to separate the use.

    train_indices: The "faiss" index of all classes for similarity search.
    test_zs_by_class: The normalized test set feature vectors that contain potentially OOD examples for each class.
    test_class_index_dict: The dictionary indicating the positions of entries from different classes.
    k: The number of considered nearst neighbors.
    """
    # Determine the number of test examples according to the normalized feature vectors by class
    nb_examples = np.sum([test_zs_by_class[uniq_class].shape[0] for uniq_class in test_zs_by_class])
    # Intialize the results
    S_total = np.zeros(nb_examples)
    # Evaluate the scores
    for uniq_class in tqdm(list(test_class_index_dict.keys()), desc='Processed classes'):
        # D represents distances, I represents index
        class_D, class_I = faiss_knn_search(train_indices[uniq_class], test_zs_by_class[uniq_class], k, display=False)
        # Evaluate the scores (S)
        class_S = -class_D[:,-1]
        # Assign the scores of the current class
        S_total[test_class_index_dict[uniq_class]] = class_S

    return S_total

def knn_OOD_detection_pipeline_sig_ver(train_indices, test_zs_by_class, test_class_index_dict, k, thresholds):
    """
    This function executes the complete k-nearst neighbor OOD detection pipeline (significant neuron version).

    train_indices: The "faiss" index of all classes for similarity search.
    test_zs_by_class: The normalized test set feature vectors that contain potentially OOD examples for each class.
    test_class_index_dict: The dictionary indicating the positions of entries from different classes.
    k: The number of considered nearst neighbors.
    thresholds: The OOD detection thresholds for different classes.
    
    Note: This version is different than the previous "by-class" version because the search index of each class contains all examples.
    """
    # Determine the number of test examples according to the normalized feature vectors by class
    nb_examples = np.sum([test_zs_by_class[uniq_class].shape[0] for uniq_class in test_zs_by_class])
    # Intialize the results
    S_total = np.zeros(nb_examples)
    # Evaluate the scores
    for uniq_class in tqdm(list(test_class_index_dict.keys()), desc='Processed classes'):       
        # D represents distances, I represents index
        class_D, class_I = faiss_knn_search(train_indices[uniq_class], test_zs_by_class[uniq_class], k, display=False)
        # Evaluate the scores (S)
        class_S = -class_D[:,-1]
        # Assign the scores of the current class
        S_total[test_class_index_dict[uniq_class]] = class_S

    # OOD detection with the given thresholds
    ood_result_total = np.zeros(S_total.shape[0]).astype(int)
    for uniq_class in test_class_index_dict:
        class_ood_result = np.less(S_total[test_class_index_dict[uniq_class]], thresholds[uniq_class]).astype(int)
        ood_result_total[test_class_index_dict[uniq_class]] = class_ood_result

    return ood_result_total

def experim_ood_detection_knn_sig_ver(train_indices, test_zs_by_class, test_class_index_dict, test_actLevels, k, train_thresholds, set_name):
    """
    This function executes the complete k-nearst neighbor OOD detection experiment.

    train_indices: The "faiss" index of all classes for similarity search.
    test_zs: The normalized test set feature vectors that contain potentially OOD examples.
    test_class_index_dict: The dictionary indicating the positions of entries from different classes.
    test_actLevels: the activation level information that contains also the original and predicted class.
    k: The number of considered nearst neighbors.
    thresholds: The OOD detection thresholds for different classes.
    set_name: The OOD set name.
    """
    # Get the ground truths and predicted class
    groundtruths = test_actLevels['class']
    predictions = test_actLevels['predict_class']
    nb_examples = groundtruths.shape[0]
    # Execute the experiment
    ood_result = knn_OOD_detection_pipeline_sig_ver(train_indices, test_zs_by_class, test_class_index_dict, k, train_thresholds)
    ood_result_bool = ood_result.astype(bool)
    # Number of InD and OOD examples and their percentages
    nb_ood = ood_result.sum()
    nb_ind = ood_result.shape[0] - nb_ood
    ood_percent = nb_ood / ood_result.shape[0]
    ind_percent = nb_ind / ood_result.shape[0]
    # Get the total, ind and ood accuracy
    acc_total = np.sum(groundtruths == predictions) / nb_examples
    acc_ood = np.sum(groundtruths[ood_result_bool] == predictions[ood_result_bool]) / nb_ood
    acc_ind = np.sum(groundtruths[np.invert(ood_result_bool)] == predictions[np.invert(ood_result_bool)]) / nb_ind
    # Display the results
    print('The number of OOD examples in', set_name, 'set:', nb_ood)
    print('The number of InD examples in', set_name, 'set:', nb_ind)
    print('The percentage of OOD examples in', set_name, 'set:', ood_percent)
    print('The percentage of InD examples in', set_name, 'set:', ind_percent)
    print('The total accuracy in', set_name, 'set:', acc_total)
    print('The accuracy on the OOD examples in', set_name, 'set:', acc_ood)
    print('The accuracy on the InD examples in', set_name, 'set:', acc_ind)

    return [set_name, nb_ood, nb_ind, ood_percent, ind_percent, acc_total, acc_ood, acc_ind]

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

def build_sig_zs(actLevels, last_hidden_layerId, sig_neuron_indices):
    """
    This function builds the normalized vectors with the significant neurons and other features as a uniteds feature.
    
    actLevels: The activation levels.
    last_hidden_layerId: The last hidden layer Id.
    sig_neuron_indices: The significant neurons indices for this layer.
    """
    # Get the number of neurons
    last_hidden_actLevels = actLevels['actLevel'][last_hidden_layerId]
    nb_neurons = last_hidden_actLevels.shape[1]
    # Determine the significant neuron indices (The sort here just to ensure that the indices are sorted)
    sorted_sig_neuron_indices = sorted(sig_neuron_indices)
    # Build the zs         
    sig_neurons = last_hidden_actLevels[:, sorted_sig_neuron_indices]
    sig_vec_norms = np.linalg.norm(sig_neurons, axis=1, keepdims=True)
    sig_zs = sig_neurons / sig_vec_norms

    return sig_zs
