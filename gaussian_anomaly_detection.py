import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def estimate_gaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariate_gaussian(dataset, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

# 小于门限概率，判定为异常点
def select_threshold(probs, test_data):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs), max(probs), stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(test_data, predictions, average='binary')
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon

    return best_f1, best_epsilon


data = pd.read_csv("./iris.data", header=None)  # header 不把第一行做为列属性
#data.columns=['sepal length','sepal width','petal length','petal width','class']

data = data.to_numpy()
x=data[:,0:4]
y=data[:,4]

X = np.array(x[: :]).astype("float")
Y = np.array(y[: :]).astype("float")

mu, sigma = estimate_gaussian(X)
p = multivariate_gaussian(X,mu,sigma)

#selecting optimal value of epsilon using cross validation
fscore, ep = select_threshold(p,Y)
print(fscore, ep)

#selecting outlier datapoints
outliers = np.asarray(np.where(p < ep))

plt.figure(1)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('Datapoints of throughput vs latency')
plt.plot(train_data[:,0], train_data[:,1],'b+')
plt.show()

plt.figure(2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('Detection of Outliers')
plt.plot(train_data[:,0],train_data[:,1],'bx')
plt.plot(train_data[outliers,0],train_data[outliers,1],'ro')
plt.show()
