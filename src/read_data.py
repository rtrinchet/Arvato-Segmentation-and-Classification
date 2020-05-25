import pickle
import pandas as pd
import feather
import joblib

from src.helpers import reduce_mem_usage

def save_dict(dictionary, dict_name):
    
    filepath = 'data/'+ dict_name + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_dict(filename):
    filepath = 'data/' + filename + '.pickle'
    print(filepath)
    
    with open(filepath, 'rb') as handle:
        loaded_dict = pickle.load(handle)

    return loaded_dict



def _logic(index, n):
    '''logic to skip rows in read_csv: keep 1 out of 5, every 5th position'''
    if index % n == 0:
        return False
    return True

def read_data(filename, sep = ';', n_skiprows = None, reduce_memory = True):
    if n_skiprows:
        data = pd.read_csv(filename , sep = sep, skiprows= lambda x: _logic(x, n_skiprows))
    else:
        data = pd.read_csv(filename , sep = sep)
        
    if reduce_memory: 
        print('Reducing memory usage...')
        data = reduce_mem_usage(data)
    print()    
    print('Completed. Shape of the data: ', data.shape)
    return data


def save_feather(data, filename):
    
    filepath = 'data/' + filename + '.ftr'
    data.to_feather(filepath)
    print(f'Data {data} saved to {filepath}')
    
    
def load_feather(filename):
    
    filepath = 'data/' + filename + '.ftr'
    
    data = pd.read_feather(filepath, columns=None, use_threads=True)
    print(f'Data loaded from to {filepath}')
    return data


def save_pipeline(pipeline, filename):
    joblib.dump(pipeline, 'models/'+filename+'.pkl')
    print(f'Pipeline saved to {filename}')
    
def load_pipeline(filename):
    pipeline = joblib.load('models/'+filename+'.pkl')
    print(f'Pipeline saved to {filename}')
    return pipeline
    
    

