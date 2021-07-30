from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


class Ensemble():
    def __init__(self) -> None:
        pass

    def predict(self, classifiers, x):
        predicoes = []
        for classificador in classifiers:
            preds = classificador.predict(x)
            predicoes.append(preds)
        predicoes = np.array(predicoes).T

        true_predictions = []

        for preds in predicoes:
            c = Counter(preds)
            true_predictions.append(c.most_common(1)[0][1])
        return true_predictions

        
def get_categorical_columns(x):
    binary = []
    nbinary = []

    for collumn in x.columns:
      if (x[collumn].dtype == 'object'):
        temp = x[collumn].unique()
        if(len(temp) == 2):
          binary.append(collumn)
        else:
          nbinary.append(collumn)

    return binary, nbinary

def process_binary_columns(x, binary):
    for collumn in binary:
      temp = x[collumn].unique()
      print("Coluna: ", collumn, "Valores: ", temp, "\n")
      i = -1
      for val in temp:
        x[collumn] = x[collumn].replace(val, i)
        i += 2

    return x

def process_nbinary_columns(x, nbinary):
    nomes = []
    for collumn in nbinary:
      temp = x[collumn].unique()
      print("Coluna: ", collumn, "Valores: ", temp, "\n")
      novas = pd.DataFrame()
      for val in temp:
        nome = collumn + '_' + val
        nomes.append(nome)
        x[nome] = np.zeros(len(x))
      for i in range(len(x)):
        x[collumn + '_' + x[collumn].iloc[i]].iloc[i] = 1
    x = x.drop(columns=nbinary)

    return x


