import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class Evaluation():
    def __init__(self, y_true, y_predict, U=None, number_of_clusters=None, n_samples=None):
        self.y_true = y_true
        self.y_predict = y_predict
        
        self.U = U
        self.number_of_clusters = number_of_clusters
        self.n_samples = n_samples
    
    # Modified Partition Entropy
    def mpc_(self, U):
        MPC = (self.number_of_clusters/(self.number_of_clusters-1)) * (1-((sum(sum(U**2)))/self.n_samples))
        return 1 - MPC

    # Partition Entropy
    def part_entropy_(self, U):
        pe = 0
        for i in range(0, self.n_samples):
            for j in range(0, self.number_of_clusters):
                L = np.log(U[j, i])
                pe += U[j, i]*L
        PE = -pe/self.n_samples
        return PE

    def print_results(self):
        corrected_rand = adjusted_rand_score(self.y_true, self.y_predict)
        f_measure = f1_score(self.y_true, self.y_predict, average='macro')
        conf_matrix = confusion_matrix(self.y_true, self.y_predict)
        acc = accuracy_score(self.y_true, self.y_predict)

        precision = 0
        recall = 0

        print('confusion matrix: ', conf_matrix)
        print('accuracy: ', acc)
        print('f-measure: ', f_measure)
        print('adjusted rand: ', corrected_rand)

        if self.U is not None:
            mpc = self.mpc_(self.U)
            p_entropy = self.part_entropy_(self.U)

            print('modified partition coef: ', mpc)
            print('partition entropy: ', p_entropy)
        else:
            precision = precision_score(self.y_true, self.y_predict, average='macro')
            recall = recall_score(self.y_true, self.y_predict, average='macro')

            print('Precision: ', precision)
            print('Recall: ', recall)

        return acc, f_measure, precision, recall
