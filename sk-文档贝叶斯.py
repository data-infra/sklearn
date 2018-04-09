import numpy as np

"""
    这个指南的目的是在一个实际任务上探索scikit-learn的主要工具，在二十个不同的主题上分析一个文本集合。
    在这一节中，可以看到：
        1、加载文本文件和类别
        2、适合机器学习的特征向量提取
        3、训练线性模型进行分类
        4、使用网格搜索策略，找到一个很好的配置的特征提取组件和分类器
"""

"""
    1、Loading the 20 newsgroups dataset 加载20个新闻组数据集
    为了获得更快的执行时间为第一个例子，我们将工作在部分数据集只有4个类别的数据集中：
"""
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print(twenty_train.target)
print(twenty_train.target_names)  # 训练集中类别的名字，这里只有四个类别
print(len(twenty_train.data))  # 训练集中数据的长度
print(len(twenty_train.filenames))  # 训练集文件名长度
print('-----')
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print('-----')
print(twenty_train.target_names[twenty_train.target[0]])
print('-----')
print(twenty_train.target[:10])  # 前十个的类别
print('-----')
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])  # 类别的名字
print('-----')
"""
    2、Extracting features from text files 从文本文件中提取特征
    为了在文本文件中使用机器学习算法，首先需要将文本内容转换为数值特征向量
"""

"""
    Bags of words 词袋
    最直接的方式就是词袋表示法
        1、为训练集的任何文档中的每个单词分配一个固定的整数ID（例如通过从字典到整型索引建立字典）
        2、对于每个文档，计算每个词出现的次数，并存储到X[i,j]中。

    词袋表示：n_features 是语料中不同单词的数量，这个数量通常大于100000.
    如果 n_samples == 10000，存储X的数组就需要10000*10000*4byte=4GB,这么大的存储在今天的计算机上是不可能实现的。
    幸运的是，X中的大多数值都是0，基于这种原因，我们说词袋是典型的高维稀疏数据集，我们可以只存储那些非0的特征向量。
    scipy.sparse 矩阵就是这种数据结构，而scikit-learn内置了这种数据结构。
"""

"""
    Tokenizing text with scikit-learn 使用scikit-learn标记文本
    文本处理、分词、过滤停用词都在这些高级组件中，能够建立特征字典并将文档转换成特征向量。
"""
from sklearn.feature_extraction.text import CountVectorizer  # sklearn中的文本特征提取组件中，导入特征向量计数函数

count_vect = CountVectorizer()  # 特征向量计数函数
X_train_counts = count_vect.fit_transform(twenty_train.data)  # 对文本进行特征向量处理
print(X_train_counts)  # 特征向量和特征标签
print(X_train_counts.shape)  # 形状
print('-----')

"""
    CountVectorizer支持计算单词或序列的N-grams，一旦合适，这个向量化就可以建立特征词典。
    在整个训练预料中，词汇中的词汇索引值与其频率有关。
"""
print(count_vect.vocabulary_.get(u'algorithm'))
print('-----')

"""
    From occurrences to frequencies 从事件到频率
    计数是一个好的开始，但是也存在一个问题：较长的文本将会比较短的文本有很高的平均计数值，即使他们所表示的话题是一样的。
    为了避免潜在的差异，它可以将文档中的每个单词出现的次数在文档的总字数的比例：这个新的特征叫做词频：tf
    tf-idf:词频-逆文档频率
"""
from sklearn.feature_extraction.text import TfidfTransformer  # sklearn中的文本特征提取组件中，导入词频统计函数

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)  # 建立词频统计函数,注意这里idf=False
print(tf_transformer)  # 输出函数属性 TfidfTransformer(norm=u'l2', smooth_idf=True, sublinear_tf=False, use_idf=False)
print('-----')
X_train_tf = tf_transformer.transform(X_train_counts)  # 使用函数对文本文档进行tf-idf频率计算
print(X_train_tf)
print('-----')
print(X_train_tf.shape)
print('-----')
"""
    在上面的例子中，使用fit()方法来构建基于数据的预测器，然后使用transform()方法来将计数矩阵用tf-idf表示。
    这两个步骤可以通过跳过冗余处理，来更快的达到相同的最终结果。
    这些可以通过使用fit_transform()方法来实现：
"""
tfidf_transformer = TfidfTransformer()  # 这里使用的是tf-idf
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf)
print(X_train_tfidf.shape)
print('-----')
"""
    Training a classifier 训练一个分类器
    既然已经有了特征，就可以训练分类器来试图预测一个帖子的类别，先使用贝叶斯分类器，贝叶斯分类器提供了一个良好的基线来完成这个任务。
    scikit-learn中包括这个分类器的许多变量，最适合进行单词计数的是多项式变量。
"""
from sklearn.naive_bayes import MultinomialNB  # 使用sklearn中的贝叶斯分类器，并且加载贝叶斯分类器

# 中的MultinomialNB多项式函数
clf = MultinomialNB()  # 加载多项式函数
x_clf = clf.fit(X_train_tfidf, twenty_train.target)  # 构造基于数据的分类器
print(x_clf)  # 分类器属性：MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print('-----')
"""
    为了预测输入的新的文档，我们需要使用与前面相同的特征提取链进行提取特征。
    不同的是，在转换中，使用transform来代替fit_transform，因为训练集已经构造了分类器
"""
docs_new = ['God is love', 'OpenGL on the GPU is fast']  # 文档
X_new_counts = count_vect.transform(docs_new)  # 构建文档计数
X_new_tfidf = tfidf_transformer.transform(X_new_counts)  # 构建文档tfidf
predicted = clf.predict(X_new_tfidf)  # 预测文档
print(predicted)  # 预测类别 [3 1]，一个属于3类，一个属于1类
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))  # 将文档和类别名字对应起来
print('-----')
"""
    Building a pipeline 建立管道
    为了使向量转换更加简单(vectorizer => transformer => classifier)，scikit-learn提供了pipeline类来表示为一个复合分类器
"""
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
print(text_clf)  # 构造分类器，分类器的属性
predicted = text_clf.predict(docs_new)  # 预测新文档
print(predicted)  # 获取预测值
print('-----')

"""
    分析总结：
        1、加载数据集，主要是加载训练集，用于对数据进行训练
        2、文本特征提取：
                对文本进行计数统计 CountVectorizer
                词频统计  TfidfTransformer  （先计算tf,再计算tfidf）
        3、训练分类器：
                贝叶斯多项式训练器 MultinomialNB
        4、预测文档：
                通过构造的训练器进行构造分类器，来进行文档的预测
        5、最简单的方式：
                通过使用pipeline管道形式，来讲上述所有功能通过管道来一步实现，更加简单的就可以进行预测
"""

"""
    Evaluation of the performance on the test set 测试集性能评价
    评估模型的预测精度同样容易：
"""
import numpy as np

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))  # 预测的值和测试值的比例，mean就是比例函数
print('-----')  # 精度已经为0.834886817577

"""
    精度已经实现了83.4%，那么使用支持向量机(SVM)是否能够做的更好呢，支持向量机(SVM)被广泛认为是最好的文本分类算法之一。
    尽管，SVM经常比贝叶斯要慢一些。
    我们可以改变学习方式，使用管道来实现分类：
"""
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline(
    [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
# _ = text_clf.fit(twenty_train.data, twenty_train.target)  # 和下面一句的意思一样，一个杠，表示本身
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))  # 精度 0.912782956059
print('-----')
"""
    sklearn进一步提供了结果的更详细的性能分析工具：
"""
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))