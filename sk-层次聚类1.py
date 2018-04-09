from time import time
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.datasets.samples_generator import make_blobs

# 产生样本数据
np.random.seed(0)

centers = [[1, 1], [-1, -1], [1, -1]]  # 三种聚类的中心
n_clusters = len(centers)
X, y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)  # 生成样本随机数


#----------------------------------------------------------------------
# 可视化聚类
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

#----------------------------------------------------------------------
# 手写体数据集
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering  # 引入层次聚类

for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)  # 通过聚类个数和聚类合并准则创建聚类模型
    begin_time = time()   # 记录开始时间
    clustering.fit(X_red)
    print(linkage,"聚类合并方法进行聚类用时: %.2fs" % (time() - begin_time))

    plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)  # 可视化聚类结果


plt.show()