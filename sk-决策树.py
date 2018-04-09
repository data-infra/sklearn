import numpy as np # 快速操作结构数组的工具
import pandas as pd # 数据分析处理工具
import matplotlib.pyplot as plt # 画图工具
from sklearn import datasets # 机器学习库
from sklearn.preprocessing import LabelEncoder
from sklearn import tree



# 下面的数据分为为每个用户的来源网站、位置、是否阅读FAQ、浏览网页数目、选择的服务类型（目标结果）
attr_arr=[['slashdot','USA','yes',18,'None'],
         ['google','France','yes',23,'Premium'],
         ['digg','USA','yes',24,'Basic'],
         ['kiwitobes','France','yes',23,'Basic'],
         ['google','UK','no',21,'Premium'],
         ['(direct)','New Zealand','no',12,'None'],
         ['(direct)','UK','no',21,'Basic'],
         ['google','USA','no',24,'Premium'],
         ['slashdot','France','yes',19,'None'],
         ['digg','USA','no',18,'None'],
         ['google','UK','no',18,'None'],
         ['kiwitobes','UK','no',19,'None'],
         ['digg','New Zealand','yes',12,'Basic'],
         ['slashdot','UK','no',21,'None'],
         ['google','UK','yes',18,'Basic'],
         ['kiwitobes','France','yes',19,'Basic']]

#生成属性数据集和结果数据集
dataMat = np.mat(attr_arr)
arrMat = dataMat[:,0:4]
resultMat = dataMat[:,4]

# 构造数据集成pandas结构，为了能理解属性的名称
attr_names = ['src', 'address', 'FAQ', 'num']   #特征属性的名称
attr_pd = pd.DataFrame(data=arrMat,columns=attr_names)    #每行为一个对象，每列为一种属性，最后一个为结果值
print(attr_pd)

#将数据集中的字符串转化为代表类别的数字。因为sklearn的决策树只识别数字
le = LabelEncoder()
for col in attr_pd.columns:                                            #为每一列序列化,就是将每种字符串转化为对应的数字。用数字代表类别
    attr_pd[col] = le.fit_transform(attr_pd[col])
print(attr_pd)

# 构建决策树
clf = tree.DecisionTreeClassifier()
clf.fit(attr_pd, resultMat)
print(clf)

# 使用决策树进行预测
result = clf.predict([[1,1,1,0]])    # 输入也必须是数字的。分别代表了每个数字所代表的属性的字符串值
print(result)

# 将决策树保存成图片
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
target_name=['None','Basic','Premium']
tree.export_graphviz(clf, out_file=dot_data,feature_names=attr_names,
                     class_names=target_name,filled=True,rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')

