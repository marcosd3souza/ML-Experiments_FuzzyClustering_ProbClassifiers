import pandas as pd
import parameters as params
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

class DataReader:
    def __init__(self):
        path = f'{params.DATA_BASE_PATH}{params.DATA_FILENAME}' 
         
        self.data = pd.read_csv(
            path, 
            delim_whitespace=True
        )

    def get_y_labels(self, y):
        temp = np.unique(y)
        
        i = 0
        for val in temp:
            y[y==val] = i
            i += 1
        return np.array(y, dtype=int)
    
    def get_data(self):
        y = np.array(self.data.values[:, -1])
        y = self.get_y_labels(y)
        X = self.data.values[:, 1:-1]
        # X = self.data.drop(columns=['class', 'sequence_name', 'pox', 'erl'])
        # X = X.values
        # X = self.scaled_data = StandardScaler().fit_transform(X)
        return np.array(X, dtype=float), y
    
    def get_oversampled_data(self, X, y):    
        sm = SMOTE(k_neighbors=3)

        # Fit the model to generate the data.
        X_new, y_new = sm.fit_resample(X, y)
        
        return X_new, y_new
    
    def get_preprocessed_data(self):
        y = np.array(self.data.values[:, -1])
        y = self.get_y_labels(y)
        X = self.data.drop(columns=['class', 'sequence_name'])

        # X_new, y_new = self.get_oversampled_data(X, y)
        
        return X.values, y
