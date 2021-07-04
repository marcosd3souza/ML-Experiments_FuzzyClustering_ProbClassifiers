from data_reader import DataReader

X, y = DataReader().get_data()

print(X.shape)
print(y.shape)