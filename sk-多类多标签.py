# 多标签分类格式。将多分类转换为二分类的格式，类似于one-hot编码
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
y_new = MultiLabelBinarizer().fit_transform(y)
print('新的输出格式：\n',y_new)



# =========1对其余的多分类构造方式================
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = LinearSVC(random_state=0)  # 构建二分类器
clf = OneVsRestClassifier(clf)  # 根据二分类器构建多分类器
clf.fit(X, y)  # 训练模型
y_pred = clf.predict(X) # 预测样本
print('预测正确的个数：%d,预测错误的个数：%d' %((y==y_pred).sum(),(y!=y_pred).sum()))


# =========1对1的多分类构造方式================
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = LinearSVC(random_state=0)  # 构建二分类器
clf = OneVsOneClassifier(clf)  # 根据二分类器构建多分类器
clf.fit(X, y)  # 训练模型
y_pred = clf.predict(X) # 预测样本
print('预测正确的个数：%d,预测错误的个数：%d' %((y==y_pred).sum(),(y!=y_pred).sum()))


# =========误差校正输出代码================
from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = LinearSVC(random_state=0)  # 构建二分类器
clf = OutputCodeClassifier(clf,code_size=2, random_state=0)  # 根据二分类器构建多分类器
clf.fit(X, y)  # 训练模型
y_pred = clf.predict(X) # 预测样本
print('预测正确的个数：%d,预测错误的个数：%d' %((y==y_pred).sum(),(y!=y_pred).sum()))


# =========多输出回归================
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
X, y = make_regression(n_samples=10, n_targets=3, random_state=1)  # 产生10个样本，每个样本100个属性，每个样本3个输出值
print('样本特征维度',X.shape)
print('样本输出维度',y.shape)
clf = GradientBoostingRegressor(random_state=0)
clf =MultiOutputRegressor(clf)
clf.fit(X, y)
y_pred = clf.predict(X) # 预测样本
print('均方误差：',metrics.mean_squared_error(y, y_pred))  # 均方误差


# =========多输出分类================
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1) # 生成分类数据集，10个样本，100个特征，30个有效特征，3种分类
y2 = shuffle(y1, random_state=1)  # 分类结果随机排序
y3 = shuffle(y1, random_state=2)  # 分类结果随机排序
Y = np.vstack((y1, y2, y3)).T  # 多种分类结果组合成
print('多输出多分类器真实输出分类:\n',Y)
n_samples, n_features = X.shape # 10,100
n_outputs = Y.shape[1] # 3个输出
n_classes = 3 # 每种输出有3种分类
forest = RandomForestClassifier(n_estimators=100, random_state=1)  # 生成随机森林多分类器
multi_target_forest = MultiOutputClassifier(forest)  # 构建多输出多分类器
y_pred = multi_target_forest.fit(X, Y).predict(X)
print('多输出多分类器预测输出分类:\n',y_pred)









