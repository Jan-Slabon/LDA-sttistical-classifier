import numpy as np
import math
import pandas as pd
class LDA:
    def __init__(self, dimension):
        self.dimension = dimension

    def learn(self,DataSet):
        self.set = np.array(DataSet)
        self.mean = np.array([np.zeros(self.dimension),np.zeros(self.dimension)], dtype='float64')
        sum=np.array(np.zeros(self.dimension),dtype='float64')
        count = 0
        for i in DataSet:
            if i[0] == 0:
                sum+=np.array(i[1:])
                count +=1
        self.mean[0] = sum/count
        sum=np.array(np.zeros(self.dimension),dtype='float64')
        count2 = 0
        for i in DataSet:
            if i[0] == 1:
                sum+=np.array(i[1:])
                count2 +=1
        self.mean[1] = sum/count2
        self.CovMatrix = np.zeros((self.dimension,self.dimension), dtype='float64')
        for i in range(self.dimension):
            for j in range(self.dimension):
                for element in DataSet:
                    el= np.array(element[1:])
                    if element[0] == 0:
                        el-=self.mean[0]
                    else:
                        el-=self.mean[1]
                    self.CovMatrix[i][j]+=el[i]*el[j]
        self.CovMatrix=self.CovMatrix/(np.size(self.set)/(self.dimension+1))

    def test(self,testing_data_source):
        results = [[0,0],[0,0]]
        testing_data = pd.DataFrame(testing_data_source)

        for row in testing_data.iterrows(): 
            if self.clasify(row[1][1:]) == row[1][0]:
                results[int(row[1][0])][0]+=1
            else: results[int(row[1][0])][1]+=1
        return results

    
    def clasify_with_parameter(self,data_src,alpha):
        data = np.array(data_src)
        #print(np.exp(-(np.dot(np.dot(data-self.mean[0],np.linalg.inv(self.CovMatrix)),data-self.mean[0]))/2)/(math.sqrt((2*3.14)**self.dimension)*math.sqrt(np.linalg.det(self.CovMatrix))))
        if np.exp(-(np.dot(np.dot(data-self.mean[0],np.linalg.inv(self.CovMatrix)),data-self.mean[0]))/2)/(math.sqrt((2*3.14)**self.dimension)*math.sqrt(np.linalg.det(self.CovMatrix))) > alpha :
            return 0
        else:
            return 1


    def clasify(self,data_src):
        data =np.array(data_src)
        if np.dot(np.dot(np.array(data),np.linalg.inv(self.CovMatrix)),(self.mean[0]-self.mean[1])) > np.dot(np.dot(self.mean[0],np.linalg.inv(self.CovMatrix)),self.mean[0])-np.dot(np.dot(self.mean[1],np.linalg.inv(self.CovMatrix)),self.mean[1])/2:
            return 0
        else:
            return 1

