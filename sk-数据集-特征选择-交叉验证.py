# -*- coding: UTF-8 -*-

import numpy as np # 快速操作结构数组的工具
import pandas as pd # 数据分析处理工具
import matplotlib.pyplot as plt # 画图工具
from sklearn import datasets # 机器学习库


# 使用鸢尾花卉样本数据，对待测对象进行分类：分别包括为山鸢尾、变色鸢尾、维吉尼亚尾


# =======加载样本数据集，清洗转化数据格式=======

#数据集 0-山鸢尾、1-变色鸢尾、2-维吉尼亚尾
scikit_iris = datasets.load_iris()     #加载鸢尾花卉数据集。每行一个对象，每列一种属性。['data']为样本数据集，['target']为结果数据集，['target_names']为类别名称，.feature_names属性名称
# 转换成pandas的DataFrame数据格式，方便观察数据
iris = pd.DataFrame(data=np.c_[scikit_iris['data'], scikit_iris['target']],columns=np.append(scikit_iris.feature_names, ['y']))    #每行为一个对象，每列为一种属性，最后一个为结果值
# print(iris.head(2))                  #查看前两行，观察数据格式
# print(iris.isnull().sum())           # isnull()返回布尔矩阵，sum()按列求和。检查数据是否有缺失
# print(iris.groupby('y').count())     # 观察样本中各类别数量是否比较均衡


# =======选择全部特征训练模型、预测新对象的分类=======

X = iris[scikit_iris.feature_names]   #获取样本集
y = iris['y']                         #获取结果集

# 第一步，选择model
from sklearn.neighbors import KNeighborsClassifier   # 导入knn分类器

knn = KNeighborsClassifier(n_neighbors=1)              # 初始化一个knn模型，设置k=1
# 第二步，fit X、y
knn.fit(X, y)                                          #根据样本集合结果集，对knn进行建模
# 第三步，predict新数据
result = knn.predict([[3, 2, 2, 5]])                   #使用knn对新对象进行预测
print(result)


# =======使用交叉验证评估模型=======
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# 分割训练-测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)  #划分训练集合测试集

# K=15
knn = KNeighborsClassifier(n_neighbors=15)                                 #创建knn模型
knn.fit(X_train, y_train)                                                  #训练knn模型

y_pred_on_train = knn.predict(X_train)       # 预测训练集，为了和预测测试集对比，查看拟合情况
y_pred_on_test = knn.predict(X_test)         # 预测测试集
# print(metrics.accuracy_score(y_train, y_pred_on_train))                      # 计算样本集的正确率
print('正确率: ：{}'.format(metrics.accuracy_score(y_test, y_pred_on_test)))  # 计算测试集的正确率
