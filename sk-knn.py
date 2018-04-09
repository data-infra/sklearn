# # ==============================无监督查找最近邻（常在聚类中使用，例如变色龙聚类算法）========================
#
# from sklearn.neighbors import NearestNeighbors
# import numpy as np # 快速操作结构数组的工具
#
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])  # 样本数据
# test_x = np.array([[-3.2, -2.1], [-2.6, -1.3], [1.4, 1.0], [3.1, 2.6], [2.5, 1.0], [-1.2, -1.3]])  # 设置测试数据
# # test_x=X  # 测试数据等于样本数据。这样就相当于在样本数据内部查找每个样本的邻节点了。
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)  # 为X生成knn模型
# distances, indices = nbrs.kneighbors(test_x)  # 为test_x中的数据寻找模型中的邻节点
# print('邻节点：',indices)
# print('邻节点距离：',distances)
#
# # ==============================使用kd树和Ball树实现无监督查找最近邻========================
#
# from sklearn.neighbors import KDTree,BallTree
# import numpy as np # 快速操作结构数组的工具
#
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# # test_x = np.array([[-3.2, -2.1], [-2.6, -1.3], [1.4, 1.0], [3.1, 2.6], [2.5, 1.0], [-1.2, -1.3]])  # 设置测试数据
# test_x=X  # 测试数据等于样本数据。这样就相当于在样本数据内部查找每个样本的邻节点了。
# kdt = KDTree(X, leaf_size=30, metric='euclidean')
# distances,indices = kdt.query(test_x, k=2, return_distance=True)
# print('邻节点：',indices)
# print('邻节点距离：',distances)



# # ==============================k最近邻分类========================
# import numpy as np # 快速操作结构数组的工具
# from sklearn.neighbors import KNeighborsClassifier,KDTree   # 导入knn分类器
#
#
# # 数据集。4种属性，3种类别
# data=[
#     [ 5.1,  3.5,  1.4,  0.2, 0],
#     [ 4.9,  3.0,  1.4,  0.2, 0],
#     [ 4.7,  3.2,  1.3,  0.2, 0],
#     [ 4.6,  3.1,  1.5,  0.2, 0],
#     [ 5.0,  3.6,  1.4,  0.2, 0],
#     [ 7.0,  3.2,  4.7,  1.4, 1],
#     [ 6.4,  3.2,  4.5,  1.5, 1],
#     [ 6.9,  3.1,  4.9,  1.5, 1],
#     [ 5.5,  2.3,  4.0,  1.3, 1],
#     [ 6.5,  2.8,  4.6,  1.5, 1],
#     [ 6.3,  3.3,  6.0,  2.5, 2],
#     [ 5.8,  2.7,  5.1,  1.9, 2],
#     [ 7.1,  3.0,  5.9,  2.1, 2],
#     [ 6.3,  2.9,  5.6,  1.8, 2],
#     [ 6.5,  3.0,  5.8,  2.2, 2],
# ]
#
# # 构造数据集
# dataMat = np.array(data)
# X = dataMat[:,0:4]
# y = dataMat[:,4]
#
# knn = KNeighborsClassifier(n_neighbors=2,weights='distance')    # 初始化一个knn模型，设置k=2。weights='distance'样本权重等于距离的倒数。'uniform'为统一权重
# knn.fit(X, y)                                          #根据样本集、结果集，对knn进行建模
# result = knn.predict([[3, 2, 2, 5]])                   #使用knn对新对象进行预测
# print(result)


# ==============================k最近邻回归========================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# 为输出值添加噪声
y[::5] += 1 * (0.5 - np.random.rand(8))

# 训练回归模型
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,weights))

plt.show()