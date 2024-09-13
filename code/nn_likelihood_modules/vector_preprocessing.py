"""
This module contains the general vector preprocessing functions
"""
"""
Imports
"""
from common_imports import *
from sklearn.impute import KNNImputer

"""
Normalization
"""
def get_mean_std(array):
    return np.mean(array), np.std(array)

def array_standardization_by_given_mean_std(array, mean, std):
    return (array - mean) / (std+1e-16)

def column_standardization_by_given_mean_std(df, column, mean, std):
    normalzied_column = array_standardization_by_given_mean_std(df[column], mean, std)
    df.loc[:, column] = normalzied_column

def array_standardization(array):
    return (array - np.mean(array)) / (np.std(array)+1e-16)

def array_min_max_normalization(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def column_standardization(df, column):
    normalzied_column = array_standardization(df[column])
    df.loc[:, column] = normalzied_column

def vectorize_column(df, column, drop=True, prefix=True):
    processed_df = df.copy(deep=True)
    if prefix:
        processed_df = pd.concat([processed_df, pd.get_dummies(processed_df[column], prefix=column)], axis=1)
    else:
        processed_df = pd.concat([processed_df, pd.get_dummies(processed_df[column])], axis=1)
    if drop:
        processed_df = processed_df.drop(column, axis=1)
    return processed_df

def vectorize_column_with_uniq_vals(df, column, uniq_vals):
    for uniq_val in uniq_vals:
        df[column+'_'+str(uniq_val)] = 0
    # Assign value
    for i, _ in df.iterrows():
        row_column_val = df.at[i,column]
        if row_column_val in uniq_vals:
            df.at[i,column+'_'+str(row_column_val)] = 1
    return df

"""
KNN imputing
"""
def knn_imputing(df, impute_headers=[], n_neighbors=20):
    """
    This function applies the knn imputing to fill the nan values (the nan value in the dataframe should be np.nan)

    df: The dataframe to impute
    impute_headers: The headers indicates the part of dataframe to be imputed
    n_neighbors: number of neighbors used to apply the imputing
    """
    # Define the used headers (if not provided in the params, we will use all features)
    if len(impute_headers) == 0:
        impute_headers = df.columns.values.tolist()
    # Imputing
    content_to_impute = df[impute_headers].to_numpy()
    imputer = KNNImputer(n_neighbors=20)
    imputed_content = imputer.fit_transform(content_to_impute)
    df[impute_headers] = imputed_content

    return df

"""
Regression filling
"""
def regression_filling(df, reg_model, data_headers, target_col):
    """
    This function fills the target column with the provided regression modes
    """
    print()
        


"""
Dataframe operations
"""
def join_two_dfs(df1, df2, ref_col):
    return pd.merge(df1, df2, on=ref_col)