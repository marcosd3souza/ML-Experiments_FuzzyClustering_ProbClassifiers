from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from classifiers.gaussian_bayesian import BayesGaussian
from classifiers.parzen_bayesian import KDEClassifier
from classifiers.ensemble import Ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv

from data_reader import DataReader
from clustering import FuzzyClustering
import time

X, y = DataReader().get_data()

t1=time.time()

best_J = 9999999
best_partition = []
best_prototypes = []
# parameters
# c = 10
# m = [1.1, 1.6, 2.0] 
# T = 150  
# e = 10^−10
m_candi = [1.1, 1.6, 2.0]
for mi in m_candi:
    fcm = FuzzyClustering(T=150, c=10, e=10**-10, m=mi)

    for it in range(0, 100):
        print(f'Iteration {it} of 100')
        J, partition, prototypes, U = fcm.fit_predict(X)

        if J is not None and J < best_J:
            best_J = J
            best_partition = partition
            best_prototypes = prototypes
            best_U = U
t2=time.time()
print("Tempo total de execução do modelo: %f" %(t2-t1))

print('############### best results ###################################')
print('J: ', best_J)
print('prototypes: ', best_prototypes)
fcm.print_results(y, best_partition, best_U)
print('################################################################')


################################################ Segunda parte do projeto ####################################################

def print_results(y_true, y_predict):
    f_measure = f1_score(y_true, y_predict, average='macro')
    conf_matrix = confusion_matrix(y_true, y_predict)
    acc = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict, average='macro')
    recall = recall_score(y_true, y_predict, average='macro')



    print('confusion matrix: ', conf_matrix)
    print('accuracy: ', acc)
    print('f-measure: ', f_measure)
    print('Precision: ', precision)
    print('Recall: ', recall)

    return acc, f_measure, precision, recall

skf = StratifiedKFold(n_splits=5)

BG = BayesGaussian()
PG = KDEClassifier()
LR = LogisticRegression()
KNN = KNeighborsClassifier()
cls = [PG, BG, LR, KNN]
classifiers_names = ['PG', 'BG', 'LR', 'KNN', 'ENSEMBLE']
ensemble = Ensemble()

path = "data\input\yeast.data"

df = pd.read_csv(path, delimiter=r"\s+")


y = df['class']
X = df.drop(columns=['class', 'sequence_name', 'vac', 'pox', 'erl'])
y = ensemble.process_class(y)


resultados = [[],[],[],[],[]]
k = 0
for train_index, test_index in skf.split(X, y):
    X_train_k, X_test_k = X.iloc[train_index], X.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]
    LR.fit(X_train_k, y_train_k)
    PG.fit(X_train_k, y_train_k)
    KNN.fit(X_train_k, y_train_k)
    BG.fit(X_train_k, y_train_k)

    predict_LR = LR.predict(X_test_k)
    predict_PG = PG.predict(X_test_k)
    predict_KNN = KNN.predict(X_test_k)
    predict_BG = BG.predict(X_test_k)

    predict_ENSEMBLE = ensemble.predict(classifiers=cls, x=X_test_k)

    print("Resultado do classificador LR para K = ", k)
    acc_LR, f_measure_LR, precision_LR, recall_LR = print_results(y_test_k, predict_LR)
    resultados[0].append([1 - acc_LR, f_measure_LR, precision_LR, recall_LR])

    print("Resultado do classificador LR para K = ", k)
    acc_PG, f_measure_PG, precision_PG, recall_PG = print_results(y_test_k, predict_PG)
    resultados[1].append([1 - acc_PG, f_measure_PG, precision_PG, recall_PG])

    print("Resultado do classificador LR para K = ", k)
    acc_KNN, f_measure_KNN, precision_KNN, recall_KNN = print_results(y_test_k, predict_KNN)
    resultados[2].append([1 - acc_KNN, f_measure_KNN, precision_KNN, recall_KNN])

    print("Resultado do classificador LR para K = ", k)
    acc_BG, f_measure_BG, precision_BG, recall_BG = print_results(y_test_k, predict_BG)
    resultados[3].append([1 - acc_BG, f_measure_BG, precision_BG, recall_BG])

    print("Resultado do classificador LR para K = ", k)
    acc_ENSEMBLE, f_measure_ENSEMBLE, precision_ENSEMBLE, recall_ENSEMBLE = print_results(y_test_k, predict_ENSEMBLE)
    resultados[4].append([1 - acc_ENSEMBLE, f_measure_ENSEMBLE, precision_ENSEMBLE, recall_ENSEMBLE])

    k += 1

n = len(y_test_k)
k = 0

metricas_friedman = {}
for name in classifiers_names:
    res = np.array(resultados[k]).T
    
    erro = np.mean(res[0])
    erro = 1.96 * np.sqrt( (erro * (1 - erro)) / n)
    print("Erro médio para " + name + " :", (erro * 100), )

    fscore = np.mean(res[1])
    fscore = 1.96 * np.sqrt( (fscore * (1 - fscore)) / n)
    print("Fscore médio para " + name + " :", (fscore))

    precision = np.mean(res[2])
    precision = 1.96 * np.sqrt( (precision * (1 - precision)) / n)
    print("Precisão média para " + name + " :", (precision * 100))

    cobertura = np.mean(res[3])
    cobertura = 1.96 * np.sqrt( (cobertura * (1 - cobertura)) / n)
    print("Recall (cobertura) média para " + name + " :", (cobertura * 100), "\n")

    metricas_friedman[name] = {"Classificador": name, "Erro": erro, "Fscore": fscore, "Precision": precision, "Recall": cobertura}
    k += 1

with open('data\output\metricas_friedman.csv', 'w') as f:
    w = csv.DictWriter(f, metricas_friedman['PG'].keys())
    w.writeheader()
    for name in classifiers_names:
        w.writerow(metricas_friedman[name])

