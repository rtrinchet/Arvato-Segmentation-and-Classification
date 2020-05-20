from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



def decide_pca_components(df, pca_pipeline, variables = True, pcts=[0.25, 0.50, 0.75, 0.9]):
    '''Supports the decision of the number of components
    Args: 
    -df: scaled dataframe
    -info: if true shows info of variance explained
    -pcts: list of pcts to show
    '''
    pca = pca_pipeline[1]
    
    if variables:
        k = df.shape[1]//10
        for i in range(5):
            n_vars = k*(i+1)
            var_exp = pca.explained_variance_ratio_[:n_vars].sum()
            print(f'Variance explained with {n_vars} variables: {var_exp}')
        print()
    
    var = np.cumsum(pca.explained_variance_ratio_)
    
    if pcts:
        for pct in pcts:
            n = len(var[var<pct])
            print(f'The {pct*100}% of explained variance is achieved with {n} components')
        
    
    # plot of Cumulative Variance explains

    plt.plot(var)
    plt.title('Variance explained by PCA number of components')
    plt.xlabel('Number of components')
    plt.ylabel('% of variance explained')
    plt.show()
    print()
    
    
def plot_PCA_weights(pca_pipeline, df, component_num, n, n_weights = 10):

    features_list = df.columns.values
    pca = pca_pipeline[1]
    
    # weights of each column in the PCA components
    v = pd.DataFrame(pca.components_)
    # index of the component 
    row_idx = n - component_num
    # getting the weights from the row as an array v_1
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    comps = pd.DataFrame(list(zip(v_1, features_list)), columns = ['weights', 'features'])
    comps['abs_weights']= comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending = False).head(n_weights)

    plt.subplots(figsize = (10,5))
    sns.barplot(data = sorted_weight_data, x = 'weights', y='features', palette = "Blues_r")
    plt.title(f'Weights for the component number {component_num}')
    plt.ylabel('')
    plt.show() 
