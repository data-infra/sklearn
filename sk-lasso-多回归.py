import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import MultiTaskLasso, Lasso

rng = np.random.RandomState(42)
# ===========================产生模拟样本数据=========================
# 用随机的频率、相位产生正弦波的二维系数
n_samples, n_features, n_tasks = 100, 30, 40  # n_samples样本个数，n_features特征个数，n_tasks估计值的个数
n_relevant_features = 5 # 自定义实际有用特征的个数
coef = np.zeros((n_tasks, n_features)) # 系数矩阵的维度

times = np.linspace(0, 2 * np.pi, n_tasks)
for k in range(n_relevant_features):
    coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1)) # 自定义数据矩阵，用来生成模拟输出值

X = rng.randn(n_samples, n_features)  # 产生随机输入矩阵
Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks) # 输入*系数+噪声=模拟输出
# ==============================使用样本数据训练系数矩阵============================
coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_  # 多任务训练

# #############################################################################
# Plot support and time series
fig = plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.spy(coef_lasso_)
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.text(10, 5, 'Lasso')
plt.subplot(1, 2, 2)
plt.spy(coef_multi_task_lasso_)
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.text(10, 5, 'MultiTaskLasso')
fig.suptitle('Coefficient non-zero location')

feature_to_plot = 0
plt.figure()
lw = 2
plt.plot(coef[:, feature_to_plot], color='seagreen', linewidth=lw,
         label='Ground truth')
plt.plot(coef_lasso_[:, feature_to_plot], color='cornflowerblue', linewidth=lw,
         label='Lasso')
plt.plot(coef_multi_task_lasso_[:, feature_to_plot], color='gold', linewidth=lw,
         label='MultiTaskLasso')
plt.legend(loc='upper center')
plt.axis('tight')
plt.ylim([-1.1, 1.1])
plt.show()