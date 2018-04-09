import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn import metrics

from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3],random_state =9)
plt.scatter(X[:, 0], X[:, 1], marker='o',c=y)
plt.show()


# 不设置聚类数目的Birch
y_pred = Birch(n_clusters = None).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print("CH指标:", metrics.calinski_harabaz_score(X, y_pred))


# 设置聚类数目的Birch
y_pred = Birch(n_clusters = 4).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print("CH指标:", metrics.calinski_harabaz_score(X, y_pred))


# 尝试多个threshold取值，和多个branching_factor取值
param_grid = {'threshold':[0.5,0.3,0.1],'branching_factor':[50,20,10]}  # 定义优化参数字典，字典中的key值必须是分类算法的函数的参数名
for threshold in param_grid['threshold']:
    for branching_factor in param_grid['branching_factor']:
        clf = Birch(n_clusters = 4,threshold=threshold,branching_factor=branching_factor)
        clf.fit(X)
        y_pred = clf.predict(X)
        print(threshold,branching_factor,"CH指标:", metrics.calinski_harabaz_score(X, y_pred))







