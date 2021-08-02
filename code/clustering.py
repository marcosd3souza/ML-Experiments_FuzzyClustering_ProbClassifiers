import numpy as np
import math

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import random
from decimal import Decimal

# Implementation of the FCM-DFCV algorithm  
class FuzzyClustering:
    def __init__(self, T: int, c: int, e: float, m: np.ndarray):
        self.number_of_iterations = T
        self.number_of_clusters = c
        self.tolerance = e
        self.m = m
    
    def get_initial_U_matrix(self, X, G):
        #Definir matriz de pertinencia initial 'U'.
        U = np.zeros([self.number_of_clusters, self.n_samples])

        for i in range(0, self.n_samples):
            for h in range(0, self.number_of_clusters):
                numerator = np.sum((X[i] - G[h])**2)
                
                result = 0
                for k in range(0, self.number_of_clusters):
                    denominator = np.sum((X[i] - G[k])**2)

                    if denominator != 0:
                        result += numerator/ (denominator ** (1/(self.m-1)))
                
                U[h, i] = 1/result if result > 0 else result
        return np.array(U)
    
    def get_U_matrix(self, X, G, M):
        # Definir matriz de pertinencia initial 'U'. (equação 20)
        U = np.ones([self.number_of_clusters, self.n_samples]) 

        for k in range(0, self.n_samples):
            for i in range(0, self.number_of_clusters):
                degree = np.float128(0.0)
                M_i_diag = np.diag(M[i])

                numerator = np.sum(M_i_diag * ((X[k] - G[i])**2), dtype=np.float128)

                for h in range(0, self.number_of_clusters):
                    M_h_diag = np.diag(M[h])
                    
                    denominator = np.sum(M_h_diag * ((X[k] - G[h])**2), dtype=np.float128)
                    if denominator != 0:
                        result = numerator/denominator
                        degree += result ** (1/(self.m-1))
                        
                if degree > 0.0:
                    U[i, k] = degree ** -1
                else:
                    U[i, k] = 1e-10

        return np.array(U)

    def update_prototypes(self, X, U):
        # update prototypes from equation (3)
        G = []
        for i in range(0, self.number_of_clusters):
            numerator = np.zeros(self.n_features)
            denominator = np.zeros(self.n_features)
            G_i = np.zeros(self.n_features)

            for k in range(0, self.n_samples):
                denominator += U[i, k] ** self.m
            
            for k in range(0, self.n_samples):
                numerator += (U[i, k] ** self.m) * X[k]
            
            if np.min(denominator) > 0.0:
                G_i = numerator / denominator
            else:
                G_i = numerator

            G.append(G_i)
        return np.array(G)

    def get_initial_prototypes(self, X: np.ndarray):
        # Matriz aleatória de representantes Gk para cada cluster k
        G = []
        check_list = []

        while len(G) < self.number_of_clusters:
            num = random.randint(0,self.n_samples-1)
            if num not in check_list:
                G.append(X[num])
                check_list.append(num)
        return np.array(G)

    def construct_M_matrix(self, X, G, U):
        M = []
        for i in range(0, self.number_of_clusters):
            Mi = np.zeros([self.n_features, self.n_features])
            Mi_diag = []
            numerator = 1.0

            for h in range(0, self.n_features):
                term = 0.0
                for k in range(0, self.n_samples):    
                    term += (U[i, k] ** self.m) * ((X[k, h] - G[i][h])**2)
                numerator *= term
            
            numerator = numerator ** (1/self.n_features)
            
            for j in range(0, self.n_features):
                denominator = 0
                for k in range(0, self.n_samples):
                    denominator += (U[i, k] ** self.m) * ((X[k, j] - G[i][j])**2)
                
                if denominator != 0:
                    result = float(numerator/denominator)
                    Mi_diag.append(result)
            
            np.fill_diagonal(Mi, Mi_diag)

            M.append(Mi)

        return np.array(M)


    def get_objective(self, X, G, U, M):
        # implement the J5 function, equation (17)
        J = 0

        for k in range(0, self.n_samples):
            for i in range(0, self.number_of_clusters):    
                v = X[k] - G[i]
                J += (U[i, k]**self.m) * np.sum(v.reshape(-1,1) * M[i] * v)

        return J
    
    def fit_predict(self, X: np.ndarray):
        self.n_samples, self.n_features = X.shape
        Js = []
        Us = []
        Gs = []
        # initial prototypes, U matrix and J
        G = self.get_initial_prototypes(X)
        U = self.get_initial_U_matrix(X, G)
        # M = self.construct_M_matrix(X, G, U)
        # U = self.get_U_matrix(X, G, M)
        # J_0 = self.get_objective(X, G_0, U_0, M)
        # print('initial obj: ', J_0)
        
        # J_old = J_0
        J_old = 99999
        
        count = 0
        for t in range(0, self.number_of_iterations):

            # print('t: ', t, 'of T:', self.number_of_iterations)
            
            G = self.update_prototypes(X, U)
            M = self.construct_M_matrix(X, G, U)           
            U = self.get_U_matrix(X, G, M)
            
            J = self.get_objective(X, G, U, M)

            # cond = len(np.unique(np.argmax(U, axis=0))) != self.number_of_clusters
            if np.isnan(J) or (J_old-J) < self.tolerance or J == 0.0:
                break
        
            Js.append(J)
            Us.append(U)
            Gs.append(G)

            J_old = J
            print('J: ', J)
        
        if len(Js) > 0:
            best_J_idx = np.argmin(Js)
            best_G = Gs[best_J_idx] 
            best_U = Us[best_J_idx]
            partition = np.argmax(Us[best_J_idx], axis=0)
            return Js[best_J_idx], partition, best_G, best_U
        else:
            return None, [], [], []

