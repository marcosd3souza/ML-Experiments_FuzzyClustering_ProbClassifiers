from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from gaussian_bayesian import BayesGaussian
from parzen_bayesian import KDEClassifier
from sklearn.neighbors import KNeighborsClassifier
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

def process_class(y, bin=False, index=0):
    temp = y.unique()
    y = y
    valoresOriginais = []
    i = 0
    for val in temp:
        if bin:
            if i == index:
                y = y.replace(val, 0)
            else:
                y = y.replace(val, 1)
        else:
            y = y.replace(val, i)
        valoresOriginais.append((val, i))
        i += 1

    return y, valoresOriginais

def best_bandwith(X,classe_train, classe_test, base_test):
    best_band_index = 0
    best_band_acc = 0
    band = 0
    arr = []
    while ( band <= 1):
        band += 0.1
        kde = KDEClassifier(bandwidth=band).fit(X, classe_train)
        c = kde.predict(base_test)
        acc_parsen = 0
        for i in range(400):
          if (classe_test[i] == c[i]):
              acc_parsen +=1
        acc_parsen = acc_parsen/400
        arr.append(acc_parsen)
        if (acc_parsen > best_band_acc):
            best_band_acc = acc_parsen
            best_band_index = band

    return best_band_index, best_band_acc, arr

path = r"C:\Users\Cejota\Documents\ML\ml-cin-2021\data\input\yeast.data"

df = pd.read_csv(path, delimiter=r"\s+")
y = df['class']
x = df.drop(columns=['class', 'sequence_name', 'vac', 'pox', 'erl'])

binary, nbinary = get_categorical_columns(x)
x = process_binary_columns(x, binary)
x = process_nbinary_columns(x, nbinary)

y, valoresOriginais = process_class(y)

X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    stratify=y,
                                                    test_size=0.25)

BG = BayesGaussian()
PG = KDEClassifier()
LR = LogisticRegression()
KNN = KNeighborsClassifier()
LR.fit(X_train, y_train)
PG.fit(X_train, y_train)
KNN.fit(X_train, y_train)
BG.fit(X_train, y_train)
cls = [PG, BG, LR, KNN]
ensemble = Ensemble()
ensemble.predict(classifiers=cls, x=X_test)

#print(np.cov(x).shape)
#BG.get_bayes_numerator(input_ex=vf, for_class=1)
#print(BG.get_conditional_prob_gaussian(1, 1, 1))
#print(BG.predict(X_test))
#BG.predict(X_test)


#LR = LogisticRegression(multi_class="ovr").fit(x, y)
