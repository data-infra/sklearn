# 产生样本数据集
from sklearn.model_selection import cross_val_score
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# # ==================Bagging 元估计器=============
# from sklearn.ensemble import BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
# scores = cross_val_score(bagging, X, y)
# print('Bagging准确率：',scores.mean())
#
# # ==================决策树、随机森林、极限森林对比===============
#
# # 决策树
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
# scores = cross_val_score(clf, X, y)
# print('决策树准确率：',scores.mean())
#
# # 随机森林
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10,max_features=2)
# scores = cross_val_score(clf, X, y)
# print('随机森林准确率：',scores.mean())
#
# # 极限随机树
# from sklearn.ensemble import ExtraTreesClassifier
# clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
# scores = cross_val_score(clf, X, y)
# print('极限随机树准确率：',scores.mean())
#
# print('模型中各属性的重要程度：',clf.feature_importances_)
#
#
# # ====================AdaBoost=========================
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators=100)
# scores = cross_val_score(clf, X, y)
# print('AdaBoost准确率：',scores.mean())
#
#
# # ====================Gradient Tree Boosting（梯度树提升）=========================
# # 分类
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
# scores = cross_val_score(clf, X, y)
# print('GDBT分类准确率：',scores.mean())
#
# # 回归
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import load_boston
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
#
# boston = load_boston()  # 加载波士顿房价回归数据集
# X1, y1 = shuffle(boston.data, boston.target, random_state=13) # 将数据集随机打乱
# X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=0)  # 划分训练集和测试集.test_size为测试集所占的比例
# clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,max_depth=4,min_samples_split=2,loss='ls')
# clf.fit(X1, y1)
# print('GDBT回归MSE：',mean_squared_error(y_test, clf.predict(X_test)))
# # print('每次训练的得分记录：',clf.train_score_)
# print('各特征的重要程度：',clf.feature_importances_)
# plt.plot(np.arange(500), clf.train_score_, 'b-')  # 绘制随着训练次数增加，训练得分的变化
# plt.show()



# ====================Voting Classifier（投票分类器）=========================

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')  # 无权重投票
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[2,1,2]) # 权重投票

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf,X,y,cv=5, scoring='accuracy')
    print("准确率: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# 配合网格搜索
from sklearn.model_selection import GridSearchCV
params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}  # 搜索寻找最优的lr模型中的C参数和rf模型中的n_estimators
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(iris.data, iris.target)
print('最优参数：',grid.best_params_)



