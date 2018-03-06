# -*- coding: UTF-8 -*-

# ========加载数据(Data Loading)========
import numpy as np
import urllib.request

# 数据集的请求地址
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# 下载响应的csv文件
raw_data = urllib.request.urlopen(url)
# 加载csv文件成numpy中的矩阵
dataset = np.loadtxt(raw_data, delimiter=",")
# 分割成属性集和结果集
X = dataset[:,0:7]   # 特征矩阵
y = dataset[:,8]  #目标矩阵
# print('特征矩阵:\n',X)
# print('结果矩阵:\n',y)

# ========数据归一化(Data Normalization)========
from sklearn import preprocessing
# 归一化数据集
normalized_X = preprocessing.normalize(X)
# 标准话数据集
standardized_X = preprocessing.scale(X)

# ========特征选择(Feature Selection)========
# 树算法(Tree algorithms)计算特征的信息量
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# 显示每个特征的重要性
print('属性重要性:\n',model.feature_importances_)

# ========逻辑回归========
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print('逻辑回归模型:\n',model)
# 使用模型预测
expected = y
predicted = model.predict(X)
# 评估模型
print(metrics.classification_report(expected, predicted))  #评估模型
print(metrics.confusion_matrix(expected, predicted))  # 使用混淆矩阵评估模型

# ========朴素贝叶斯========
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
print('朴素贝叶斯模型:\n',model)
# 使用模型预测
expected = y
predicted = model.predict(X)
# 评估模型
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# ========k近邻========
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# 使用样本数据构建knn模型
model = KNeighborsClassifier()
model.fit(X, y)
print('KNN模型:\n',model)
# 使用模型预测
expected = y
predicted = model.predict(X)
# 评估模型
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# ========决策树========
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# 构建决策树模型
model = DecisionTreeClassifier()
model.fit(X, y)
print('决策树模型:\n',model)
# 使用模型预测
expected = y
predicted = model.predict(X)
# 评估模型
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# ========支持向量机========
from sklearn import metrics
from sklearn.svm import SVC
# 构建svm模型
model = SVC()
model.fit(X, y)
print('SVM模型:\n',model)
# 使用模型预测
expected = y
predicted = model.predict(X)
# 评估模型
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# ========优化算法参数========
import numpy as np
from sklearn.linear_model import Ridge   #岭回归模型
from scipy.stats import uniform as sp_rand
from sklearn.grid_search import GridSearchCV  #网格搜索
from sklearn.grid_search import RandomizedSearchCV  # 随机搜索

# 准备参数的可取值
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
# 构建岭回归模型，并尝试参数每一个可取值
model = Ridge()
rsearch = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))

# # 只给定区间，参数随机取值
# param_grid = {'alpha': sp_rand()}
# # 构建岭回归模型，并尝试参数随机值
# model = Ridge()
# rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)

rsearch.fit(X, y)
print(rsearch)
# 评估搜索结果
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)
