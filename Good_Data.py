import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import LDA as clasifier
firstmean = [-1,3,-2]
seconmean = [2,5,8]
variance = [[7,0,0],[0,5,0],[0,0,6]]
tab = []
tmp1 = []
tmp2 = []
tmp3 = []
for i in range(500):
    tmp = np.array(np.random.multivariate_normal(firstmean,variance))
    tab.append([0,tmp[0],tmp[1],tmp[2]]) 
    tmp1.append(tmp[0])
    tmp2.append(tmp[1])
    tmp3.append(tmp[2])
for i in range(500):
    tmp = np.array(np.random.multivariate_normal(seconmean,variance))
    tab.append([1,tmp[0],tmp[1],tmp[2]])
    tmp1.append(tmp[0])
    tmp2.append(tmp[1])
    tmp3.append(tmp[2])



lda =  clasifier.LDA(3)
lda.learn(tab)
print(lda.mean[0])
print(lda.mean[1])
print(lda.CovMatrix)


ax = plt.axes(projection='3d')
ax.scatter3D(tmp1[500:],tmp2[500:],tmp3[500:],marker='x',color='red')
ax.scatter3D(tmp1[0:500],tmp2[0:500],tmp3[0:500],marker='o',color='blue')
plt.show()


testing_data = []

for i in range(100):
    tmp = np.random.binomial(1,0.5)
    if tmp == 1:
        random_vector = np.array(np.random.multivariate_normal(firstmean,variance))
        testing_data.append([0,random_vector[0],random_vector[1],random_vector[2]])
    else:
        random_vector = np.array(np.random.multivariate_normal(seconmean,variance))
        testing_data.append([1,random_vector[0],random_vector[1],random_vector[2]])





results = lda.test(testing_data)
sensivity=(results[0][0]/(results[0][0]+results[0][1])) #precision 0
specificity=(results[1][0]/(results[1][0]+results[1][1])) # precision 1
accuracy = (results[1][0]+results[0][0])/(results[1][0]+results[0][0]+results[1][1]+results[0][1]) # dokladnosc
recalA = results[0][0]/(results[0][0]+results[1][1])
recalB = results[1][0]/(results[1][0]+results[0][1])
F1A = 2*(sensivity*recalA)/(sensivity+recalA)
F1B = 2*(specificity*recalB)/(specificity+recalB)
print(results)
print("czułość", sensivity)
print("swoistość", specificity)
print("dokładność", accuracy)
print(F1A)
print(F1B)


