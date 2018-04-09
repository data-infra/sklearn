from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  -2.],
                    [ 2.,  0.,  0.],
                    [ 3.,  1., -1.]])
X_test = [[-1., 1., 0.]]


# ===============标准化====================
# 计算数据集的尺度（也就是数据集的均值和方差）（各列）
scaler = preprocessing.StandardScaler().fit(X_train)   # 计算均值和方差
print('均值：',scaler.mean_ )
print('方差：',scaler.scale_ )

# 通过尺度去处理另一个数据集，当然另一个数据集仍然可以是自己。
X_scaled = scaler.transform(X_train)
print('均值：',X_scaled.mean(axis=0))  # transform会转化数据集为均值为0
print('方差：',X_scaled.std(axis=0))   # transform会转化数据集为方差为1

# 上面两步的综合：缩放样本，是样本均值为0，方差为1（各列）
X_scaled = preprocessing.scale(X_train,axis=0)      # 标准化：去均值和方差
print('均值：',X_scaled.mean(axis=0))
print('方差：',X_scaled.std(axis=0))

# =====================特征缩放====================
# MinMaxScaler将特征缩放至特定范围内（默认为0-1）
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)  # 训练同时转换
print('每列最大值：',X_train_minmax.max(axis=0))   # 每列最大值为1
print('每列最小值：',X_train_minmax.min(axis=0))    # 每列最小值为0
# 缩放对象是记录了，平移距离和缩放大小，再对数据进行的操作
print('先平移：',min_max_scaler.min_)
print('再缩放：',min_max_scaler.scale_)

X_test_minmax = min_max_scaler.transform(X_test)   # 转换实例应用到测试数据:实现和训练数据一致的缩放和移位操作:



# MaxAbsScaler通过除以每个特征的最大值将训练数据特征缩放至 [-1, 1] 范围内。可以应用在稀疏矩阵上保留矩阵的稀疏性。
X_train = np.array([[ 0., -1.,  0.],
                    [ 0., 0.,  0.2],
                    [ 2.,  0., 0]])
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
print('每列最大值：',X_train_maxabs.max(axis=0))   # 每列最大值为1
print('每列最小值：',X_train_maxabs.min(axis=0))    # 每列最小值不低于-1
print('缩放比例：',max_abs_scaler.scale_)
X_test_maxabs = max_abs_scaler.transform(X_test)   # 转换实例应用到测试数据:实现和训练数据一致的缩放和移位操作:
print('缩放后的矩阵仍然具有稀疏性：\n',X_train_maxabs)



# ===================缩放有离群值的数据========================
X_train = np.array([[ 1., -11.,  -2.],
                    [ 2.,  2.,  0.],
                    [ 13.,  1., -11.]])
robust_scale = preprocessing.RobustScaler()
X_train_robust = robust_scale.fit_transform(X_train)  # 训练同时转换
print('缩放后的矩阵离群点被处理了：\n',X_train_maxabs)




# ===================非线性转换===================
X_train = np.array([[ 1., -1.,  -2.],
                    [ 2.,  0.,  0.],
                    [ 3.,  1., -1.]])
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)  # 将数据映射到了零到一的均匀分布上（默认是均匀分布）
X_train_trans = quantile_transformer.fit_transform(X_train)

#查看分位数信息，经过转换以后，分位数的信息基本不变
print('源分位数情况：',np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
print('变换后分位数情况：',np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))

# 下面将数据映射到了零到一的正态分布上：输入的中值称为输出的平均值，并且以0为中心。正常输出被剪切，使得输入的最小和最大值分别对应于1e-7和1-1e-7分位数
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',random_state=0)


X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
# ===================样本归一化===================
X_normalized = preprocessing.normalize(X, norm='l1')  # 使用 l1 或 l2 范式。缩放使每个样本（每行）的一范数或二范数为1
print('样本归一化：\n',X_normalized)
# 当然仍然可以先通过样本获取转换对象，再用转换对象归一化其他数据
normalizer = preprocessing.Normalizer().fit(X)  # 获取转换对象
normalizer.transform(X)  # 转换任何数据，X或测试集

# ===================特征二值化===================
binarizer = preprocessing.Binarizer().fit(X)  # 获取转换模型，生成的门限，默认为0
print(binarizer)
# binarizer = preprocessing.Binarizer(threshold=1) # 自定义转换器。门限以上为1，门限（包含）以下为0
X_normalized = binarizer.transform(X)  # 转换任何数据，X或测试集
print('特征二值化：\n',X_normalized)



# ===================分类特征编码(one-hot编码)===================
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 1, 2],   # 每列一个属性，每个属性一种编码
         [1, 0, 0],
         [0, 2, 1],
         [1, 0, 1]])
print('取值范围整数个数：',enc.n_values_)  # 每个属性的最大可取值数目。2,3,3
print('编码后：',enc.transform([[0, 1, 1]]).toarray()) # 转换目标对象。根据可取值所占位数进行罗列。前2位为第一个数字one-hot编码，紧接着的3位为第二个数字的编码，最后3位为第三个数字的编码
print('特征开始位置的索引：',enc.feature_indices_) # 对 n_values_的累积值，代表一个样本转换为编码后的每个属性的开始位置。0,2,5,8


# ===================缺失值插补===================
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)  # missing_values参数设定的值被认为是缺失值，计算均值时忽略不计
imp.fit([[1, 2],   # 计算每列的非空值的均值
         [np.nan, 3],
         [7, 6]])

X = [[np.nan, 2], [6, np.nan], [7, 6]]
print('缺失值插值后：\n',imp.transform(X))  # 使用每个的均值为每列缺失值插补


# ===================生成多项式特征===================
from sklearn.preprocessing import PolynomialFeatures
X = np.array([[0, 1],
              [2, 3],
              [4, 5]])
poly = PolynomialFeatures(2,interaction_only=False)  # 最大二次方。interaction_only参数设置为True，则会只保留交互项
print('生成多项式：\n',poly.fit_transform(X))   # 从 (X_1, X_2) 转换为 (1, X_1, X_2, X_1^2, X_1X_2, X_2^2)

