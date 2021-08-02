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

from evaluation import Evaluation
from data_reader import DataReader
from clustering import FuzzyClustering
import time

X, y = DataReader().get_preprocessed_data() # get_data()

n_samples = X.shape[0]
number_of_clusters = 10

fcm_results = {
    'mpc': [],
    'p_entropy': [],
    'cr': [],
    'acc': [],
    'f_measure': []
}

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
    fcm = FuzzyClustering(T=150, c=number_of_clusters, e=10**-10, m=mi)

    for it in range(0, 100):
        print(f'Iteration {it} of 100')
        J, partition, prototypes, U = fcm.fit_predict(X)

        metrics = Evaluation(y, partition, U, number_of_clusters, n_samples)
        mpc, p_entropy, acc, f_measure, corrected_rand = metrics.get_clustering_results()

        fcm_results['mpc'].append(mpc)
        fcm_results['p_entropy'].append(p_entropy)
        fcm_results['cr'].append(corrected_rand)
        fcm_results['acc'].append(acc)
        fcm_results['f_measure'].append(f_measure)

        if J is not None and J < best_J:
            best_J = J
            best_partition = partition
            best_prototypes = prototypes
            best_U = U
t2=time.time()
print("Tempo total de execução do modelo: %f" %(t2-t1))

df = pd.DataFrame(fcm_results)
df.to_csv('./../data/output/clustering_results.csv', sep=';', index=False)

print('############### best results ###################################')
print('J: ', best_J)
print('prototypes: ', best_prototypes)
# fcm.print_results(y, best_partition, best_U)

clustering_metrics = Evaluation(y, best_partition, best_U, number_of_clusters, n_samples)
clustering_metrics.print_results()

print('################################################################')

################################################ Segunda parte do projeto ####################################################

skf = StratifiedKFold(n_splits=5)

BG = BayesGaussian()
PG = KDEClassifier()
LR = LogisticRegression()
KNN = KNeighborsClassifier()
cls = [PG, BG, LR, KNN]
classifiers_names = ['PG', 'BG', 'LR', 'KNN', 'ENSEMBLE']
ensemble = Ensemble()

folds_results = {
    'fold': [],
    'classifier': [],
    'error_rate': [],
    'f_measure': [],
    'precision': [],
    'recall': []
}
resultados = [[],[],[],[],[]]
k = 0
for train_index, test_index in skf.split(X, y):
    X_train_k, X_test_k = X[train_index], X[test_index]
    y_train_k, y_test_k = y[train_index], y[test_index]
    
    LR.fit(X_train_k, y_train_k)
    PG.fit(X_train_k, y_train_k)
    KNN.fit(X_train_k, y_train_k)
    BG.fit(X_train_k, y_train_k)

    y_predict_LR = LR.predict(X_test_k)
    LR_metrics = Evaluation(y_test_k, y_predict_LR)

    y_predict_PG = PG.predict(X_test_k)
    PG_metrics = Evaluation(y_test_k, y_predict_PG)

    y_predict_KNN = KNN.predict(X_test_k)
    KNN_metrics = Evaluation(y_test_k, y_predict_KNN)

    y_predict_BG = BG.predict(X_test_k)
    BG_metrics = Evaluation(y_test_k, y_predict_BG)
    
    y_predict_ENSEMBLE = ensemble.predict(classifiers=cls, x=X_test_k)
    ENSEMBLE_metrics = Evaluation(y_test_k, y_predict_ENSEMBLE)

    print("Resultado do classificador LR para K = ", k)
    acc_LR, f_measure_LR, precision_LR, recall_LR = LR_metrics.print_results()
    resultados[0].append([1 - acc_LR, f_measure_LR, precision_LR, recall_LR])
    folds_results['fold'].append(k)
    folds_results['classifier'].append('LR')
    folds_results['error_rate'].append(1 - acc_LR)
    folds_results['f_measure'].append(f_measure_LR)
    folds_results['precision'].append(precision_LR)
    folds_results['recall'].append(recall_LR)

    print("Resultado do classificador PG para K = ", k)
    acc_PG, f_measure_PG, precision_PG, recall_PG = PG_metrics.print_results()
    resultados[1].append([1 - acc_PG, f_measure_PG, precision_PG, recall_PG])
    folds_results['fold'].append(k)
    folds_results['classifier'].append('PG')
    folds_results['error_rate'].append(1 - acc_PG)
    folds_results['f_measure'].append(f_measure_PG)
    folds_results['precision'].append(precision_PG)
    folds_results['recall'].append(recall_PG)

    print("Resultado do classificador KNN para K = ", k)
    acc_KNN, f_measure_KNN, precision_KNN, recall_KNN = KNN_metrics.print_results()
    resultados[2].append([1 - acc_KNN, f_measure_KNN, precision_KNN, recall_KNN])
    folds_results['fold'].append(k)
    folds_results['classifier'].append('KNN')
    folds_results['error_rate'].append(1 - acc_KNN)
    folds_results['f_measure'].append(f_measure_KNN)
    folds_results['precision'].append(precision_KNN)
    folds_results['recall'].append(recall_KNN)

    print("Resultado do classificador BG para K = ", k)
    acc_BG, f_measure_BG, precision_BG, recall_BG = BG_metrics.print_results()
    resultados[3].append([1 - acc_BG, f_measure_BG, precision_BG, recall_BG])
    folds_results['fold'].append(k)
    folds_results['classifier'].append('BG')
    folds_results['error_rate'].append(1 - acc_BG)
    folds_results['f_measure'].append(f_measure_BG)
    folds_results['precision'].append(precision_BG)
    folds_results['recall'].append(recall_BG)

    print("Resultado do classificador ENSEMBLE para K = ", k)
    acc_ENSEMBLE, f_measure_ENSEMBLE, precision_ENSEMBLE, recall_ENSEMBLE = ENSEMBLE_metrics.print_results()
    resultados[4].append([1 - acc_ENSEMBLE, f_measure_ENSEMBLE, precision_ENSEMBLE, recall_ENSEMBLE])
    folds_results['fold'].append(k)
    folds_results['classifier'].append('ENSEMBLE')
    folds_results['error_rate'].append(1 - acc_ENSEMBLE)
    folds_results['f_measure'].append(f_measure_ENSEMBLE)
    folds_results['precision'].append(precision_ENSEMBLE)
    folds_results['recall'].append(recall_ENSEMBLE)

    k += 1

df = pd.DataFrame(folds_results)
df.to_csv('./../data/output/folds_results.csv', sep=';', index=False)

n = len(y_test_k)
k = 0

metricas_avg = {}
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

    metricas_avg[name] = {"Classificador": name, "Erro": erro, "Fscore": fscore, "Precision": precision, "Recall": cobertura}
    k += 1

# with open('data\output\metricas_friedman.csv', 'w') as f:
#     w = csv.DictWriter(f, metricas_friedman['PG'].keys())
#     w.writeheader()
#     for name in classifiers_names:
#         w.writerow(metricas_friedman[name])

