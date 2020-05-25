
import numpy as np
import pandas as pd

from src.helpers import *

def variable_deletion(df):    
    print('Initial df shape:', df.shape)
    # variables to drop
    vars_to_drop = []
    
    print('Variables with missing values...')
    missing_df = missing_values_table(df)
    to_drop_missing = list(missing_df.loc[missing_df['% of Total Values'] > 50 ].index) # select the columns with more than half missing

    vars_to_drop.extend(to_drop_missing)
    print(f'\tDropped {len(to_drop_missing)} variables')


    print('Highly correlated variables...')

    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
    k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.8
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
    vars_to_drop.extend(to_drop_corr)
    print(f'\tDropped {len(to_drop_corr)} variables')

    print('Constant variables')
    to_drop_constant = df.columns[df.nunique() <= 1]
    vars_to_drop.extend(to_drop_constant)

    print(f'\tDropped {len(to_drop_constant)} variables')

    df_clean = df.drop(vars_to_drop, axis = 1)
    print('Final df shape:', df_clean.shape)
    
    
    return df_clean, vars_to_drop


def categorical_clean(df):
    
    df.CAMEO_DEU_2015.replace({'XX':np.NaN}, inplace=True)
    
    df.CAMEO_DEUG_2015.replace({'X':np.NaN}, inplace=True)
    df.CAMEO_DEUG_2015 = df.CAMEO_DEUG_2015.astype(float)
    
    df.CAMEO_INTL_2015.replace({'XX':np.NaN}, inplace=True)
    df.CAMEO_INTL_2015 = df.CAMEO_INTL_2015.astype(float)
    
    return df



def clean_data(df):
    # index
    df = df.set_index('LNR')
    
    df_clean, vars_to_drop = variable_deletion(df)
    
    df_clean = df_clean.drop('EINGEFUEGT_AM', axis = 1)
        
    df_clean = categorical_clean(df_clean)
    
    return df_clean


    