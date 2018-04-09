from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score,cross_validate  # 交叉验证中的模型度量
import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.metrics import make_scorer
from sklearn import metrics

#  =============================分类度量===============================
print('=============================分类度量===============================')
iris = datasets.load_iris()  # 加载iris 数据集；用于分类问题
X, y = iris.data, iris.target  # 150个样本，4个属性，3种分类


clf = svm.SVC(probability=True, random_state=0)

# ===========================交叉验证获取度量=======================
score = cross_val_score(clf, X, y, scoring='accuracy',cv=3)  # 默认进行三次交叉验证
print('交叉验证度量：',score)


# ===========================自定义度量=======================

# 自定义度量函数，输入为真实值和预测值
def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)

loss  = make_scorer(my_custom_loss_func, greater_is_better=False) # 自定义度量对象。结果越小越好。greater_is_better设置为false，系统认为是损失函数，则会将计分函数取反
score = make_scorer(my_custom_loss_func, greater_is_better=True) # 自定义度量对象。结果越大越好
clf = svm.SVC()
clf.fit(X, y)

print(loss(clf,X,y)) # 对模型进行度量，系统会自动调用模型对输入进行预测，并和真实输出值进行比较，计算损失函数
print(score(clf,X,y)) # 对模型进行度量，系统会自动调用模型对输入进行预测，并和真实输出值进行比较，计算得分


# ============================多种度量值=========================
scoring = ['precision_macro', 'recall_macro'] # precision_macro为精度，recall_macro为召回率
scores = cross_validate(clf, X, y,scoring=scoring,cv=5, return_train_score=True)
sorted(scores.keys())
print('多种度量的测试结果：',scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）



# ============================分类指标=========================
clf = svm.SVC()  # 构建模型
clf.fit(X, y) # 训练模型
predict_y = clf.predict(X) # 预测数据

print('准确率指标：',metrics.accuracy_score(y, predict_y))  # 计算准确率
print('Kappa指标：',metrics.cohen_kappa_score(y, predict_y)) # Kappa 检验
print('混淆矩阵：\n',metrics.confusion_matrix(y, predict_y)) # 混淆矩阵

target_names = ['class 0', 'class 1', 'class 2']
print('分类报告：\n',metrics.classification_report(y, predict_y, target_names=target_names))  # 分类报告
print('汉明损失：',metrics.hamming_loss(y, predict_y))  #汉明损失 。在多分类中， 汉明损失对应于 y 和 predict_y 之间的汉明距离
print('Jaccard 相似系数：',metrics.jaccard_similarity_score(y, predict_y))   # Jaccard 相似系数



# 下面的系数在在二分类中不需要使用average参数，在多分类中需要使用average参数进行多个二分类的平均
# average可取值：macro（宏）、weighted（加权）、micro（微）、samples（样本）、None（返回每个类的分数）

print('精度计算：',metrics.precision_score(y, predict_y, average='macro'))
print('召回率：',metrics.recall_score(y, predict_y,average='micro'))
print('F1值：',metrics.f1_score(y, predict_y,average='weighted'))

print('FB值：',metrics.fbeta_score(y, predict_y,average='macro', beta=0.5))
print('FB值：',metrics.fbeta_score(y, predict_y,average='macro', beta=1))
print('FB值：',metrics.fbeta_score(y, predict_y,average='macro', beta=2))
print('精确召回曲线：',metrics.precision_recall_fscore_support(y, predict_y,beta=0.5,average=None))
print('零一损失：',metrics.zero_one_loss(y, predict_y))

# ROC曲线(二分类)
y1 = np.array([0, 0, 1, 1])  # 样本类标号
y_scores = np.array([0.1, 0.4, 0.35, 0.8]) # 样本的得分（属于正样本的概率估计、或置信度值）
fpr, tpr, thresholds = metrics.roc_curve(y1, y_scores, pos_label=1)
print('假正率：',fpr)
print('真正率：',tpr)
print('门限：',thresholds)
print('AUC值：',metrics.roc_auc_score(y1, y_scores))


labels = np.array([0, 1, 2])  # 三种分类的类标号
pred_decision = clf.decision_function(X)  # 计算样本属于每种分类的得分，所以pred_decision是一个3列的矩阵
print('hinge_loss：',metrics.hinge_loss(y, pred_decision, labels = labels))

# 逻辑回归损失，对真实分类和预测分类概率进行对比的损失
y_true = [0, 0, 1, 1]
y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
print('log_loss：',metrics.log_loss(y_true, y_pred))


# ===============================回归度量==============================
print(' ===============================回归度量==============================')
diabetes = datasets.load_diabetes()  # 加载糖尿病数据集；用于回归问题
X, y = diabetes.data, diabetes.target  # 442个样本，10个属性，数值输出

model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(X, y)   # 线性回归建模
predicted_y = model.predict(X)  # 使用模型预测

print('解释方差得分：',metrics.explained_variance_score(y, predicted_y))  # 解释方差得分
print('平均绝对误差：',metrics.mean_absolute_error(y, predicted_y))  # 平均绝对误差
print('均方误差：',metrics.mean_squared_error(y, predicted_y))  # 均方误差
print('均方误差对数：',metrics.mean_squared_log_error(y, predicted_y))  # 均方误差对数
print('中位绝对误差：',metrics.median_absolute_error(y, predicted_y))  # 中位绝对误差
print('可决系数：',metrics.r2_score(y, predicted_y, multioutput='variance_weighted')) #可决系数
print('可决系数：',metrics.r2_score(y, predicted_y, multioutput='raw_values')) #可决系数
print('可决系数：',metrics.r2_score(y, predicted_y, multioutput='uniform_average')) #可决系数







