import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

# #############################################################################
# 产生样本数据
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]  # 三种聚类的中心
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)  # 生成样本随机数

# #############################################################################
# k均值聚类

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
begin_time = time.time()  # 记录训练开始时间
k_means.fit(X) # 聚类模型
t_batch = time.time() - begin_time  # 记录训练用时
print('k均值聚类时长：',t_batch)
# #############################################################################
# 小批量k均值聚类
# batch_size为每次更新使用的样本数
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
begin_time = time.time()  # 记录训练开始时间
mbk.fit(X) # 聚类模型
t_mini_batch = time.time() -  begin_time  # 记录训练用时
print('小批量k均值聚类时长：',t_mini_batch)
# #############################################################################
# 结果可视化
fig = plt.figure(figsize=(16, 6))  # 窗口大小
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)  # # 窗口四周留白
colors = ['#4EACC5', '#FF9C34', '#4E9A06']  # 三种聚类的颜色

# 在两种聚类算法中，样本的所属类标号和聚类中心
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0) # 三个聚类点排序
mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0) # 三个聚类点排序
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers) # 计算X中每个样本与k_means_cluster_centers中的哪个样本最近。也就是获取所有对象的所属的类标签
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers) # 计算X中每个样本与k_means_cluster_centers中的哪个样本最近。也就是获取所有对象的所属的类标签
order = pairwise_distances_argmin(k_means_cluster_centers,mbk_means_cluster_centers)  # 计算k均值聚类点相对于小批量k均值聚类点的索引。因为要比较两次聚类的结果的区别，所以类标号要对应上


# 绘制KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k  # 获取属于当前类别的样本
    cluster_center = k_means_cluster_centers[k]  # 获取当前聚类中心
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',markerfacecolor=col, marker='.') # 绘制当前聚类的样本点
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6) # 绘制聚类中心点
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))

# 绘制MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == k # 获取属于当前类别的样本
    cluster_center = mbk_means_cluster_centers[k] # 获取当前聚类中心
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',markerfacecolor=col, marker='.') # 绘制当前聚类的样本点
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6) # 绘制聚类中心点
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %(t_mini_batch, mbk.inertia_))

# 初始化两次结果中
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)

for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == order[k]))  # 将两种聚类算法中聚类结果不一样的样本设置为true，聚类结果相同的样本设置为false

identic = np.logical_not(different)  # 向量取反，也就是聚类结果相同设置true，聚类结果不相同设置为false

ax.plot(X[identic, 0], X[identic, 1], 'w',markerfacecolor='#bbbbbb', marker='.') # 绘制聚类结果相同的样本点
ax.plot(X[different, 0], X[different, 1], 'w',markerfacecolor='m', marker='.') # 绘制聚类结果不同的样本点
ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())

plt.show()