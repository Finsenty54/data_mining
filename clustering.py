#!/user/bin/python3
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import datasets

data=pd.read_csv("iris.data",header=None) #header 不把第一行做为列属性
#data.columns=['sepal length','sepal width','petal length','petal width','class']

data=data.to_numpy()
#各取出10条，共30条
train=np.vstack((data[0:10,:],data[50:60,:],data[100:110,:]))

print("训练数据：\n",train)
X=train[:,0:4] #data
Y=train[:,4] #target

estimators = KMeans(n_clusters=3,init='random')
estimators.fit(X)



