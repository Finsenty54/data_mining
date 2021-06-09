import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans


np.random.seed(42)

# 读取数据
data = pd.read_csv("iris.data", header=None)  # header 不把第一行做为列属性
#data.columns=['sepal length','sepal width','petal length','petal width','class']

data = data.to_numpy()

train = data
train = np.array(train)
X = train[:, 0:4]
Y = train[:, 4]

Y_num = []
for i in Y:
    if i == "Iris-setosa":
        Y_num.append(0)
    elif i == "Iris-versicolor":
        Y_num.append(1)
    else:
        Y_num.append(2)

Y_num = np.array(Y_num)

Kmeans = KMeans(n_clusters=3)
result = Kmeans.fit_predict(X)
sum_err_0 = 0
sum_err_1 = 0
sum_err_2 = 0
for i in range(len(result)):
    if result[i] == 0:
        sum_err_0 += 1
    elif result[i] == 1:
        sum_err_1 += 1
    elif result[i] == 2:
        sum_err_2 += 1

sum_err=abs(50-sum_err_0)+abs(50-sum_err_1)+abs(50-sum_err_2)

print("未进行异常检测前错误率", sum_err/len(result))

##############################################
# 建立模型
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# 寻找异常点
y_pred = clf.fit_predict(X)
# print(y_pred)
X_scores = clf.negative_outlier_factor_

results = {"[sepal length]": train[:, 0],
           "[sepal width]": train[:, 1],
           "[petal length]": train[:, 2],
           "[petal width]": train[:, 3],
           "[Class]": train[:, 4],
           "[异常检测结果]": y_pred}

pd.set_option('display.max_rows', None)
results = DataFrame(results)
# print(results)
# row_ = []
# for i in range(len(y_pred)):
#     if y_pred[i] == -1:
#         row_.append(i)

# print(row_)
# print(X)

row_de = y_pred != -1
# print(row_de)

# 用boolean indexing方法删去异常点
# y_de = y_pred
# y_de=y_de[row_de]
X_de_lof = X[row_de]

y_de_lof = Y_num[row_de]

# np.delete(X, row_, axis=0)
# np.delete(Y, row_, axis=0)
# np.delete(y_de, row_, axis=0)


# deleted = {"class": Y, "abno": y_de}
# deleted = DataFrame(deleted)
# print(deleted)


Kmeans_de = KMeans(n_clusters=3)
result_de = Kmeans_de.fit_predict(X_de_lof)
# print(result_de)
sum_err_de = 0
for i in range(len(result_de)):
    if result_de[i] != y_de_lof[i]:
        sum_err_de += 1

print(result_de)
print("LOF异常检测后错误率", sum_err_de/len(result_de))

##########################################

gaussian=EllipticEnvelope(contamination=outliers_fraction)
result=gaussian.fit(X).predict(X)
print(result)