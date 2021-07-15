from logistic_regression import LogisticRegressionClassifier
import pandas as pd

path = r"C:\Users\Cejota\Documents\ML\ml-cin-2021\data\input\yeast.data"

df = pd.read_csv(path, delimiter=r"\s+")
print(df)
y = df['class']
x = df.drop('class', 1)

#TODO: Processar os dados categoricos

lgr = LogisticRegressionClassifier(x.shape[1])
lgr.fit(x, y, learning_rate = 0.02, iterations = 500)


#parameters_out = train(x, y, learning_rate = 0.02, iterations = 500)
#print(parameters_out)
