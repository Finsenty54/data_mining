#!/user/bin/python3
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data=pd.read_csv("iris.data",header=None) #header 不把第一行做为列属性
#data.columns=['sepal length','sepal width','petal length','petal width','class']

data=data.to_numpy()
train=np.vstack((data[0:30,:],data[50:80,:],data[100:130,:]))

print("训练数据：\n",train)
X=train[:,0:4]
Y=train[:,4]

#print(X)
#需要列
#Y =np.vstack((np.reshape(data[0:30,4],(30,1)),np.reshape(data[50:80,4],(30,1)),np.reshape(data[100:130,4],(30,1)))).ravel()
#Y=np.append(data[0:30,4],data[50:80,4],data[100:130,4])
#print(Y)

#SVM 多类 one-versus-one 方式 ， 产生 n_classes * (n_classes - 1) / 2 个分类器
#“one-vs-rest” 产生 n 个分类器

clf = SVC(kernel='rbf', gamma=0.7, C=1).fit(X, Y)


pre=np.vstack((data[30:35,:],data[80:85,:],data[130:135,:]))

print("\n测试数据：\n",pre)
#同样需要列
pre_x=pre[:,0:4]
pre_label=pre[:,4]

#,data[130:135,4]))
#print(pre_label)

predictions=clf.predict(pre_x)

print("\n训练结果:\n")
A=0
for idx, prediction, label in zip(enumerate(pre_x), predictions, pre_label):
    if prediction!=label:
        A+=1
        print(idx, '原', prediction, '错误识别为', label) 

print("\n分类错误率： " ,A/15)

#dec = clf.decision_function([[1]]) #返回n_classes * (n_classes-1) / 2

#print( dec.shape[1])