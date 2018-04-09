# # ======================PCA主成分分析=================
# # 花卉样本数据集
# from sklearn import datasets
# import matplotlib.pyplot as plt
# import numpy as np
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#

# from sklearn.decomposition import PCA,IncrementalPCA   # 主成分分析（PCA）
# pca = PCA(n_components=2)  # PCA降维到2维
# X_pca = pca.fit_transform(X)
#
# ipca = IncrementalPCA(n_components=2, batch_size=10)  # 增量PCA降维到2维
# X_ipca = ipca.fit_transform(X)
#
# pca = PCA(n_components=2, svd_solver='randomized', whiten=True)  # PCA 使用随机SVD
# X_pca1 = pca.fit_transform(X)
#
#
# # 绘制PCA降维后的显示
# plt.subplot(131)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=.8, lw=2)
# plt.title('PCA')
#
# # 绘制增量PCA降维后的显示
# plt.subplot(132)
# plt.scatter(X_ipca[:, 0], X_ipca[:, 1], c=y, alpha=.8, lw=2)
# plt.title('IPCA')
#
# # 绘制PCA使用随机SVD降维后的显示
# plt.subplot(133)
# plt.scatter(X_pca1[:, 0], X_pca1[:, 1], c=y, alpha=.8, lw=2)
# plt.title('PCA with rand SVD')
# plt.show()







# ======================核PCA主成分分析=================
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import numpy as np
X, y = make_circles(n_samples=400, factor=.3, noise=.05)  # 生成样本数据集

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)  # 核PCA降维
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)

pca = PCA(n_components=2)  # PCA降维到2维
X_pca = pca.fit_transform(X)

# # 绘制原始数据
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.8, lw=2)
plt.title('Original space')

# 绘制PCA降维后的显示
plt.subplot(222)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=.8, lw=2)
plt.title('PCA')

# 绘制KPCA降维后的显示
plt.subplot(223)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, alpha=.8, lw=2)
plt.title('KPCA')

# 绘制逆空间的显示
plt.subplot(224)
plt.scatter(X_back[:, 0], X_back[:, 1], c=y, alpha=.8, lw=2)
plt.title('inverse space')

plt.show()







# # ======================SparsePCA 稀疏主成分分析=================




# # =================隐 Dirichlet 分配=================
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 隐 Dirichlet 分配
# lda = LinearDiscriminantAnalysis(n_components=2)  # 降维到2维
# X_r2 = lda.fit(X, y).transform(X)
#
# # Percentage of variance explained for each components
# print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
#
#
#
# plt.subplot(122)
# for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#     plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,label=target_name)
#
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('LDA of IRIS dataset')
#
# plt.show()