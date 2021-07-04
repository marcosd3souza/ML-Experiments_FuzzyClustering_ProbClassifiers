import pandas as pd
import parameters as params

class DataReader:
    def __init__(self):
        path = f'{params.DATA_BASE_PATH}{params.DATA_FILENAME}' 
         
        self.data = pd.read_csv(
            path, 
            header=None, 
            delim_whitespace=True
        )
    
    def get_data(self):
        y = self.data.values[:, -1]
        X = self.data.values[:, 1:-1]
        return X, y
