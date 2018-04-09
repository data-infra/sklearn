from sklearn.datasets import load_iris  # 自带的样本数据集
from sklearn.neighbors import KNeighborsClassifier  # 要估计的是knn里面的参数，包括k的取值和样本权重分布方式
import matplotlib.pyplot as plt  # 可视化绘图
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV  # 网格搜索和随机搜索

iris = load_iris()

X = iris.data  # 150个样本，4个属性
y = iris.target # 150个类标号

k_range = range(1, 31)  # 优化参数k的取值范围
weight_options = ['uniform', 'distance']  # 代估参数权重的取值范围。uniform为统一取权值，distance表示距离倒数取权值
# 下面是构建parameter grid，其结构是key为参数名称，value是待搜索的数值列表的一个字典结构
param_grid = {'n_neighbors':k_range,'weights':weight_options}  # 定义优化参数字典，字典中的key值必须是分类算法的函数的参数名
print(param_grid)

knn = KNeighborsClassifier(n_neighbors=5)  # 定义分类算法。n_neighbors和weights的参数名称和param_grid字典中的key名对应


# ================================网格搜索=======================================
# 这里GridSearchCV的参数形式和cross_val_score的形式差不多，其中param_grid是parameter grid所对应的参数
# GridSearchCV中的n_jobs设置为-1时，可以实现并行计算（如果你的电脑支持的情况下）
grid = GridSearchCV(estimator = knn, param_grid = param_grid, cv=10, scoring='accuracy') #针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
grid.fit(X, y)

print('网格搜索-度量记录：',grid.cv_results_)  # 包含每次训练的相关信息
print('网格搜索-最佳度量值:',grid.best_score_)  # 获取最佳度量值
print('网格搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('网格搜索-最佳模型：',grid.best_estimator_)  # 获取最佳度量时的分类器模型


# 使用获取的最佳参数生成模型，预测数据
knn = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'], weights=grid.best_params_['weights'])  # 取出最佳参数进行建模
knn.fit(X, y)  # 训练模型
print(knn.predict([[3, 5, 4, 2]]))  # 预测新对象



# =====================================随机搜索===========================================
rand = RandomizedSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_iter=10, random_state=5)  #
rand.fit(X, y)

print('随机搜索-度量记录：',grid.cv_results_)  # 包含每次训练的相关信息
print('随机搜索-最佳度量值:',grid.best_score_)  # 获取最佳度量值
print('随机搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('随机搜索-最佳模型：',grid.best_estimator_)  # 获取最佳度量时的分类器模型


# 使用获取的最佳参数生成模型，预测数据
knn = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'], weights=grid.best_params_['weights'])  # 取出最佳参数进行建模
knn.fit(X, y)  # 训练模型
print(knn.predict([[3, 5, 4, 2]]))  # 预测新对象


# =====================================自定义度量===========================================
from sklearn import metrics
# 自定义度量函数
def scorerfun(estimator, X, y):
    y_pred = estimator.predict(X)
    return metrics.accuracy_score(y, y_pred)

rand = RandomizedSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_iter=10, random_state=5)  #
rand.fit(X, y)

print('随机搜索-最佳度量值:',grid.best_score_)  # 获取最佳度量值





