# -*- coding: UTF-8 -*-

import numpy as np # 快速操作结构数组的工具
from sklearn import svm  # svm支持向量机
import matplotlib.pyplot as plt # 可视化绘图


data_set = np.loadtxt("SVM_data.txt")
train_data = data_set[:,0:2]   # 训练特征空间
train_target = np.sign(data_set[:,2])  # 训练集类标号

test_data = [[3,-1], [1,1], [7,-3], [9,0]] # 测试特征空间
test_target = [-1, -1, 1, 1]  # 测试集类标号

plt.scatter(data_set[:,0],data_set[:,1],c=data_set[:,2])  # 绘制可视化图
plt.show()


clf = svm.SVC()  # 创建模型，参数均使用默认值
clf.fit(train_data, train_target)  # 训练模型
result = clf.predict(test_data)  # 使用模型预测值
print(result)  # 输出预测值[-1. -1.  1.  1.]


# # ===============================Linear SVM======================
from sklearn.svm import LinearSVC

clf = LinearSVC() # 创建线性可分svm模型，参数均使用默认值
clf.fit(train_data, train_target)  # 训练模型
result = clf.predict(test_data)  # 使用模型预测值
print(result)  # 输出预测值[-1. -1.  1.  1.]