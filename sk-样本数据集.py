# import numpy as np # 快速操作结构数组的工具
# import pandas as pd # 数据分析处理工具
# import matplotlib.pyplot as plt # 画图工具
# from sklearn import datasets # 机器学习数据集
# from sklearn.datasets import make_blobs
# from sklearn import datasets

# load_boston([return_X_y]) 加载波士顿房价数据；用于回归问题
# load_iris([return_X_y]) 加载iris 数据集；用于分类问题
# load_diabetes([return_X_y]) 加载糖尿病数据集；用于回归问题
# load_digits([n_class, return_X_y]) 加载手写字符集；用于分类问题
# load_linnerud([return_X_y]) 加载linnerud 数据集；用于多元回归问题


# # ===========房价数据===========
# from sklearn.datasets import load_boston
# from sklearn import linear_model
# boston = load_boston()
# data=boston.data
# target = boston.target
# print(data.shape)
# print(target.shape)
#
# print('系数矩阵:\n',linear_model.LinearRegression().fit(data,target).coef_)
#
#
# # ===========花卉数据===========
# from sklearn.datasets import load_iris
# from sklearn import svm
# iris = load_iris()
# data=iris.data
# target = iris.target
# print(data.shape)
# print(target.shape)
#
# print('svm模型:\n',svm.SVC().fit(data,target))

# # ===========糖尿病数据集===========
# from sklearn.datasets import load_diabetes
# from sklearn import linear_model
# diabetes = load_diabetes()
# data=diabetes.data
# target = diabetes.target
# print(data.shape)
# print(target.shape)
#
# print('系数矩阵:\n',linear_model.LinearRegression().fit(data,target).coef_)



# # ===========手写体数据===========
# from sklearn.datasets import load_digits
# import matplotlib.pyplot as plt # 画图工具
# digits = load_digits()
# data=digits.data
# print(data.shape)
# plt.matshow(digits.images[3])  # 矩阵像素点的样式显示3
# # plt.imshow(digits.images[3])  # 图片渐变的样式显示3
# # plt.gray()   # 图片显示为灰度模式
# plt.show()


# #  # ===========多元回归===========
# from sklearn.datasets import load_linnerud
# from sklearn import linear_model
# linnerud = load_linnerud()
# data=linnerud.data
# target = linnerud.target
# print(data.shape)
# print(target.shape)
#
# print('系数矩阵:\n',linear_model.LinearRegression().fit(data,target).coef_)



# # ===========图像样本数据集===========
# from sklearn.datasets import load_sample_image
# import matplotlib.pyplot as plt # 画图工具
# img=load_sample_image('flower.jpg')   # 加载sk自带的花朵图案
# plt.imshow(img)
# plt.show()



# # ===========生成分类样本数据集===========
# from sklearn import datasets
# import matplotlib.pyplot as plt # 画图工具
# data,target=datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,n_repeated=0, n_classes=2, n_clusters_per_class=1)
# print(data.shape)
# print(target.shape)
# plt.scatter(data[:,0],data[:,1],c=target)
# plt.show()


# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.datasets import make_blobs
# from sklearn.datasets import make_gaussian_quantiles
# from sklearn.datasets import make_hastie_10_2
#
# plt.figure(figsize=(10, 10))
# plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
#
# plt.subplot(421)
# plt.title("One informative feature, one cluster per class", fontsize='small')
# X1, Y1 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1,n_clusters_per_class=1)
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
#
# plt.subplot(422)
# plt.title("Two informative features, one cluster per class", fontsize='small')
# X1, Y1 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1)
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
#
# plt.subplot(423)
# plt.title("Two informative features, two clusters per class", fontsize='small')
# X2, Y2 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2)
# plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)
#
# plt.subplot(424)
# plt.title("Multi-class, two informative features, one cluster",fontsize='small')
# X1, Y1 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1, n_classes=3)
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
#
# plt.subplot(425)
# plt.title("Three blobs", fontsize='small')
# # 1000个样本，2个属性，3种类别，方差分别为1.0,3.0,2.0
# X1, Y1 = make_blobs(n_samples=1000, n_features=2, centers=3,cluster_std=[1.0,3.0,2.0])
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
#
# plt.subplot(426)
# plt.title("Gaussian divided into four quantiles", fontsize='small')
# X1, Y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=4)
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
#
# plt.subplot(427)
# plt.title("hastie data ", fontsize='small')
# X1, Y1 = make_hastie_10_2(n_samples=1000)
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
# plt.show()



# # ===========生成圆形或月亮型分类数据===========

from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

fig = plt.figure(1)
x1, y1 = make_circles(n_samples=1000, factor=0.5, noise=0.1)
plt.subplot(121)
plt.title('make_circles function example')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)

plt.subplot(122)
x1, y1 = make_moons(n_samples=1000, noise=0.1)
plt.title('make_moons function example')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)
plt.show()



# # =======清洗转化数据格式======
# # 转换成pandas的DataFrame数据格式，方便观察数据
# pddata = pd.DataFrame(data=np.c_[data, target],columns=np.append(['x1','x2'], ['y']))    #每行为一个对象，每列为一种属性，最后一个为结果值
# # print(iris.head(2))                  #查看前两行，观察数据格式
# # print(iris.isnull().sum())           # isnull()返回布尔矩阵，sum()按列求和。检查数据是否有缺失
# # print(iris.groupby('y').count())     # 观察样本中各类别数量是否比较均衡
