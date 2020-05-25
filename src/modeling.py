import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler 


from sklearn.decomposition import PCA

import lightgbm as lgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.feature_selection import var_lists

SEED = 0

def create_pipeline(df, model):
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop = 'first'))
    ])

    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    
    numerical_columns, categorical_columns = var_lists(df)

    preprocessing = ColumnTransformer(
        [('cat', categorical_pipe, categorical_columns),
         ('num', numerical_pipe, numerical_columns)])
    
    classifier_pipe = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', model)
    ])

    return classifier_pipe


def create_pipeline_PCA(df, pca_model, classifier):
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop = 'first'))
    ])

    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    numerical_columns, categorical_columns = var_lists(df)

    preprocessing = ColumnTransformer(
        [('cat', categorical_pipe, categorical_columns),
         ('num', numerical_pipe, numerical_columns)])
    
    classifier_pipe = Pipeline([
        ('preprocess', preprocessing),
        ('pca', pca_model),
        ('classifier', classifier)
    ])

    return classifier_pipe



def create_pipeline_smote(df, classifier, k):
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop = 'first'))
    ])

    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    numerical_columns, categorical_columns = var_lists(df)

    preprocessing = ColumnTransformer(
        [('cat', categorical_pipe, categorical_columns),
         ('num', numerical_pipe, numerical_columns)])
    
    over = RandomOverSampler(sampling_strategy=0.03, random_state = SEED)
    under = RandomUnderSampler(sampling_strategy=0.7, random_state = SEED)
    
    clf = Pipeline([('preprocessing', preprocessing),
                    ('over', over),
                    ('under', under),
                    ('classifier', classifier)])

    return clf



def determine_interactions(X_train, y_train, interaction_vars, threshold = 0.005):

    df = X_train[interaction_vars].copy()
    
    model = create_pipeline(X_train[interaction_vars], lgb.LGBMClassifier(random_state = SEED))
    baseline = cross_val_score(model, df, y_train, cv=StratifiedKFold(5), scoring = 'roc_auc').mean()
    
    interactions = list()
    used = list()
    for feature_A in interaction_vars:
        used.append(feature_A)
        for feature_B in set(interaction_vars) - set(used):
            df['interaction_1'] = df[feature_A] * df[feature_B]
            df['interaction_2'] = df[feature_A] + df[feature_B]
            df['interaction_3'] = df[feature_A] - df[feature_B]

            model = create_pipeline(df, lgb.LGBMClassifier(random_state = SEED))
            score = cross_val_score(model, df, y_train, cv=StratifiedKFold(5), scoring = 'roc_auc').mean()
            if score > baseline + threshold:
                interactions.append((feature_A, feature_B, round(score,3)))

    print(f'Baseline: {baseline}')
    print()
    print('Top 10 interactions')

    interactions_df = pd.DataFrame(interactions, columns=['feature_A', 'feature_B', 'metric']).sort_values(by = 'metric', ascending = False)
    print(interactions_df.head())
    
    return interactions_df




def create_fi(df, var_list, feats_a, feats_b):
    
    df_ini = df.copy()
    df = df[var_list]
    for feature_A in feats_a:
        df[feature_A+'^2'] = df[feature_A]**2

        for feature_B in feats_b:
            df[feature_B+'^2'] = df[feature_B]**2

            df[feature_A +' * '+ feature_B] = df[feature_A] * df[feature_B]
            df[feature_A +' + '+ feature_B] = df[feature_A] + df[feature_B]
            df[feature_A +' - '+ feature_B] = df[feature_A] - df[feature_B]
            
    df_fi = pd.concat([df_ini[var_list], df], axis = 1)
    
    print(df_ini.shape)
    print(df_fi.shape)
    return df_fi







def create_submission(test, model, filename, index):
    test_clean = model[0].fit_transform(test)
    sub = model['classifier'].predict_proba(test_clean)[:,1]
    
    submission = pd.DataFrame({'LNR': index, 'RESPONSE':sub})
    submission.to_csv('data/' + filename, index = False)
    
    print(f"Submission stored in {'data/' + filename}")
    