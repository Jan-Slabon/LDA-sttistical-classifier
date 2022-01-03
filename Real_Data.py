import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LDA as clasifier



data = pd.read_csv("Fish.csv")
print(data)
data1 = data.loc[data['Species']=='Bream']
data2 = data.loc[data['Species']=='Perch']

data3 = pd.concat([data1,data2])

data3['Class'] = (data3['Species']=='Bream').astype(int)

cols = list(data3.columns.values)
data3 = data3[[cols[-1]]+cols[1:6]]



learning_data = data3.sample(frac=0.2)
print(learning_data)

tmp1 = set([tuple(elem) for elem in learning_data.values])
tmp2 = set([tuple(elem) for elem in data3.values])

LDA = clasifier.LDA(5)
LDA.learn(learning_data.to_numpy())

testing_data = pd.DataFrame(list(tmp2.difference(tmp1))) 

sensivity = []
specificity = []

results = LDA.test(testing_data)
sensivity.append(results[0][0]/(results[0][0]+results[0][1]))
specificity.append((results[1][0]/(results[1][0]+results[1][1])))


print(results)
print("czułość", sensivity)
print("swoistość", specificity)
print(LDA.mean)



