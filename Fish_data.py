from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import logistic_regresion as lr
import statistics as stat



data = pd.read_csv("Fish.csv")
data1 = data.loc[data['Species']=='Bream']
data2 = data.loc[data['Species']=='Perch']

data3 = pd.concat([data1,data2])
mean =np.mean(data3["Weight"])
var = stat.stdev(data3["Weight"])
data3["Weight"] = (data3["Weight"]-mean)/var
mean =np.mean(data3["Length1"])
var = stat.stdev(data3["Length1"])
data3["Length1"] = (data3["Length1"]-mean)/var
mean =np.mean(data3["Length2"])
var = stat.stdev(data3["Length2"])
data3["Length2"] = (data3["Length2"]-mean)/var
mean =np.mean(data3["Length3"])
var = stat.stdev(data3["Length3"])
data3["Length3"] = (data3["Length3"]-mean)/var
data3['Class'] = (data3['Species']=='Bream').astype(int)

cols = list(data3.columns.values)

data3 = data3[[cols[-1]]+cols[1:7]]


learning_data = data3.sample(frac=0.5)

tmp1 = set([tuple(elem) for elem in learning_data.values])
tmp2 = set([tuple(elem) for elem in data3.values])
Result = lr.log_reg(learning_data.to_numpy(), 6)
Clasifier = Result[0]
testing_data = pd.DataFrame(list(tmp2.difference(tmp1))).to_numpy() 

sensivity = []
specificity = []
results = []
for i in testing_data:
    tmp = np.array(i)
    results.append(int(Clasifier(tmp[1:])>=0.5 and tmp[0]==1 or Clasifier(tmp[1:])<=0.5 and tmp[0]==0))
print(sum(results)/size(results))
print(results)


