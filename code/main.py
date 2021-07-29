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