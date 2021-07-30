import numpy as np
import pandas as pd
import sys
from numpy.linalg import inv
from numpy.linalg import det
import math

class BayesGaussian():
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.num_features = 8
        self.ccps_likelihoods = None

    def get_class_priors(self, y_target):
         self.classes = list(set(y_target))
         self.class_priors = {Class: np.count_nonzero(y_target == Class) / len(y_target) for Class in self.classes}
         return self.class_priors


    def get_ccps_likelihoods(self, X_features, y_target):
        self.num_features = X_features.shape[1]
        self.ccps_likelihoods = {}
        for Class in self.classes:
            current_class_only = np.array([X_features.iloc[row_num, :] for row_num in range(len(y_target)) if y_target.iloc[row_num] == Class])
            self.ccps_likelihoods[Class] = [None for feature_num in range(self.num_features)]
            for feature_num in range(self.num_features):
                current_feature_intersection_class = current_class_only[:, feature_num]
                self.ccps_likelihoods[Class][feature_num] = {"class": Class,
                                                            "feature_number": feature_num,
                                                            "mean": np.mean(current_feature_intersection_class),
                                                            "variance": np.var(current_feature_intersection_class, ddof=0)}
        return self.ccps_likelihoods
    
    def fit(self, X_features, y_target):
        self.get_class_priors(y_target=y_target)
        self.get_ccps_likelihoods(X_features=X_features, y_target=y_target)

        return self

    def get_conditional_prob_gaussian(self, for_class, x):
        covM = []
        fmean = []
        det = 1.0
        for i in range(len(x)):
            val = (self.ccps_likelihoods[for_class][i]["variance"])
            if val != 0:
                covM.append(1/val)
            else:
                covM.append(0)
            fmean.append(self.ccps_likelihoods[for_class][i]["mean"])
            det *= val



        term1 = x - fmean
        term2 = term1 * covM
        term3 = term1[np.newaxis]
        term4 = np.dot((-(1/2)) * term3, term2)

        term5 = np.exp(term4)

        term7 = ((2.0 * math.pi) ** (- len(x) /2 ))
        term8 = det * term5
        term9 = term7 * term8
        return term9

    def get_bayes_numerator(self, input_ex, for_class):
        p_feature_given_class = self.get_conditional_prob_gaussian(for_class=for_class,
                                                                    x=input_ex
                                                                    )
        p_class = self.class_priors[for_class]
        bayes_numerator = p_feature_given_class * p_class

        return bayes_numerator

    def get_bayes_denominador(self, input_ex):
        total = 0
        for i in range(len(self.classes)):
            p_feature_given_class = self.get_conditional_prob_gaussian(for_class=i,
                                                                    x=input_ex
                                                                    )
            p_class = self.class_priors[i]
            deno = p_feature_given_class * p_class
            total += deno
        return total

    def predict_one(self, x):
        preds = []
        denominador = self.get_bayes_denominador(x)
        for i in range(len(self.classes)):
            numerador = self.get_bayes_numerator(for_class=i,
                                                input_ex=x
                                                )
            preds.append(numerador / denominador)
        return np.argmax(preds)

    def predict(self, x):
        predicoes = []
        for i in range(len(x)):
            predicoes.append(self.predict_one(x.iloc[i].values))
        return np.array(predicoes)
