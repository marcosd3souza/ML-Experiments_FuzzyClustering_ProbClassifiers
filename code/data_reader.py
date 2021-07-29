import pandas as pd
import parameters as params
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
        X = self.scaled_data = MinMaxScaler().fit_transform(X)
        return np.array(X, dtype=float), y
