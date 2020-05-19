import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_auc_score

import logging

SEED = 0

def split(df, SEED = SEED):
    X = df.drop('Target', axis = 1)
    y = df.Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=SEED)

    logging.info(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def var_lists(df):
    
    categorical_columns = list(df.select_dtypes('object').columns)
    numerical_columns = list(set(df.columns) - set(categorical_columns))
    if 'Target' in numerical_columns: numerical_columns.remove('Target')
    
    logging.info(f'Numerical columns: {len(numerical_columns)}. Categorical columns: {len(categorical_columns)}')
    
    return numerical_columns, categorical_columns

def get_feats(model, cat_cols, num_cols): 
    '''Fitted rf as input'''
    
    logging.info('Categorical columns: ', cat_cols)
    
    ohe = (model.named_steps['preprocess']
         .named_transformers_['cat']
         .named_steps['onehot'])
    
    feature_names = ohe.get_feature_names(input_features=cat_cols)
    feature_names = np.r_[feature_names, num_cols]
    
    return feature_names

def create_pipe(categorical_columns, numerical_columns):
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop = 'first'))
    ])
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessing = ColumnTransformer(
        [('cat', categorical_pipe, categorical_columns),
         ('num', numerical_pipe, numerical_columns)])

    rf = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', RandomForestClassifier(random_state=SEED))
    ])
    
    return rf




def create_num_pipe(numerical_columns):
    
    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    rf = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', RandomForestClassifier(random_state=SEED))
    ])
    
    return rf

def whole_pipeline(df):
    
    
    X_train, X_test, y_train, y_test = split(df)
    
    numerical_columns, categorical_columns = var_lists(df)
    
    rf = create_pipe(categorical_columns, numerical_columns)

    rf.fit(X_train, y_train)

    logging.info("Model train accuracy: %0.3f" % rf.score(X_train, y_train))
    logging.info("Model test accuracy: %0.3f" % rf.score(X_test, y_test))
    y_pred = rf.predict(X_test)
    
    feats = get_feats(rf, categorical_columns, numerical_columns)
    
    print(f"Pipeline completed. Model ROC-AUC: {roc_auc_score(y_test, y_pred):0.3f}.")
    logging.info('{:=^60}'.format(' Pipeline info '))
    logging.info(f"Input df shape: {df.shape}. Number of features after preprocessing: {len(feats)}.")
    logging.info(f"No. of categorical columns: {len(categorical_columns)}. No. of numerical columns: {len(numerical_columns)}" )
    
    return rf, feats



def test_performance(df, reduced_list, verbose = None):
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
    
    logging.info('First part: all variables included\n')
    # first part: all variables included!
    model, feats = whole_pipeline(df)
    
    logging.info('='*30)
    
    # performance with reduced vars
    logging.info(f'Second part: list of {len(reduced_list)} variables included\n')
    
    numerical_columns, categorical_columns = var_lists(df)
    feats = get_feats(model, categorical_columns, numerical_columns)
    X_train, X_test, y_train, y_test = split(df)

    # change train-test: reduce variables
    X_train = pd.DataFrame(model['preprocess'].fit_transform(X_train), columns = feats)
    X_train_reduced = X_train[reduced_list]
    
    X_test = pd.DataFrame(model['preprocess'].fit_transform(X_test), columns = feats)
    X_test_reduced = X_test[reduced_list]
    
    logging.info('Xtrain done!',X_train.shape)
    logging.info(X_train_reduced.shape)
    logging.info('Xtest done!', X_test.shape)
    logging.info(X_test_reduced.shape)
    
    # create new model: just take care of numerical variables
    # as now X_train_reduced will only have the numerical variables resulting of the first
    # pass of the pipeline
    model = create_num_pipe(reduced_list)

    logging.info('model fitting...')
    model.fit(X_train_reduced, y_train)
    logging.info('Fit done! Printing results...')
    logging.info("Model train accuracy with reduced variables: %0.3f" % model.score(X_train_reduced, y_train))
    logging.info("Model test accuracy with reduced variables: %0.3f" % model.score(X_test_reduced, y_test))
    y_pred = model.predict(X_test_reduced)
    final_result = roc_auc_score(y_test, y_pred)
    print(f"Model with {len(reduced_list)} variables succesfully trained. Final ROC-AUC: {final_result :0.3f}")
    
    return final_result
    

