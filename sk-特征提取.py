# ===========从字典类型加载特征。形成系数矩阵结构==========
from sklearn.feature_extraction import DictVectorizer
measurements = [
    {'name': 'student1', 'age': 12},
    {'boy':True, 'parents': 'baba'},
    {'size':16},
]

vec = DictVectorizer().fit(measurements)  # 定义一个加载器，后对一个字典对象提取特征。（值为数值型、布尔型的属性为单独的属性。值为字符串型的属性，形成"属性=值"的新属性）
print('提取的特征：',vec.get_feature_names())  # 查看提取的新属性
print('稀疏矩阵形式：\n',vec.transform(measurements))
print('二维矩阵形式：\n',vec.transform(measurements).toarray())

# =================文本特征提取==============
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?',]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)  # 默认提取至少 包含2个字母的单词
print('所有特征：',vectorizer.get_feature_names())
print('样本特征向量：\n',X.toarray())  # X本身为稀疏矩阵存储形式，toarray转换为二维矩阵形式

print('document属性的列索引：',vectorizer.vocabulary_.get('document'))  # 从 特征 名称到矩阵的（列索引）

# 提取一个单词或两个单词形成的词组。这样就能识别“is this”和“this is”这两种词汇了
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print('所有分词：',analyze('Bi-grams are cool!'))






