#!/user/bin/python3
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

'''
oneDim = np.array([1.0,2,3,4,5])
twoDim = np.array([[1,2],[3,4],[5,6],[7,8]]) 
arrFromTuple = np.array([(1,'a',3.0),(2,'b',3.5)])
'''

data=pd.read_csv("iris.data",header=None) #header 不把第一行做为列属性
data.columns=['sepal length','sepal width','petal length','petal width','class']
#print(data.head())
'''
for col in data.columns:
    if is_numeric_dtype(data[col]):
        print('%s:' % (col)) #取字符串
        print('\t Mean =%.2f' % data[col].mean())
        print('\t Standard deviation = %.2f' % data[col].std())
        print('\t Minimum = %.2f' % data[col].min())
        print('\t Maximum = %.2f' % data[col].max())

print(data['class'].value_counts())

print(data.describe(include='all'))

print('Covariance:')#协方差
print(data.cov())

print('Correlation:')
print(data.corr())
'''

'''
x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')  # Plot some data on the (implicit) axes.
plt.plot(x, x**2, label='quadratic')  # etc.
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
'''

#plt.figure(figsize=(12,9))
'''
plt.subplot(131)
data['sepal length'].hist(bins=8)
'''

#fig,axes=plt.subplots(1,2)

#data.boxplot()
#data.plot(kind='box',subplots=True,layout=(3,4),sharex=True,sharey=True)
cla=data['class'].values
#print(cla)
data['Class']=pd.Series(cla)
data.boxplot(by='Class',figsize=(12,12))#,ax=axes[0]
'''plt.subplot(133)

plt.scatter(x="sepal length",y="petal length",data=data)
plt.xlabel("sepal length")
plt.ylabel("pepal length")
'''
#plt.show()

c_cla=[]
for da in cla:
    if da=='Iris-setosa':
        c_cla.append('red')
    elif da=='Iris-versicolor':
        c_cla.append('blue')
    else:
        c_cla.append('green')
#print(c_cla)
scatter_matrix(data,label=cla,c=c_cla,diagonal='kde',figsize=(12,12))
plt.text(0,0,'Iris-setosa',c='red',ha='center',va='center')
plt.text(1,0,'Iris-versicolor',c='blue',ha='center',va='center')
plt.text(2,0,'Iris-virginica',c='green',ha='center',va='center')

#axes[1].legend(ncol=3, loc='lower left', fontsize='small')

plt.show()


'''
category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    print(data)
    data_cum = data.cumsum(axis=1) #返回沿给定轴的元素的累加和。axis=1 sum over columns for each of the  rows
    #计算每行的列总和https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html#numpy.cumsum
    #结果是行上每列有前几项的累加值
    category_colors = plt.get_cmap('RdYlGn')( #Get a colormap instance, defaulting to rc values if name is None.
        np.linspace(0.15, 0.85, data.shape[1])) #获取5种颜色

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    result_zip=list(zip(category_names,category_colors))
    print(result_zip)
    for i, (colname, color) in enumerate(zip(category_names, category_colors)): #目录颜色匹配，enum获得索引
        widths = data[:, i]#获取每列数据，行是选择所有，列只选取一项，matlab一样
        starts = data_cum[:, i] - widths#得到起始点
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)): #取中间位置
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)
plt.show()
'''


'''
url = "pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
data.hist()
plt.show()

'''



















'''
def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out

data1, data2, data3, data4 = np.random.randn(4, 100)
fig, ax = plt.subplots(3, 2,figsize=(12,12))
my_plotter(ax, data1, data2, {'marker': 'x'})
fig.show()
'''
