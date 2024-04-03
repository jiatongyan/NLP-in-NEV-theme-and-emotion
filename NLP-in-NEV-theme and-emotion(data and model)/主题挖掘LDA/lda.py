import os
import pandas as pd
import re
import jieba
import jieba.posseg as psg

output_path = r'D:\pythonProject\lda\result'
file_path = r'D:\pythonProject\lda\data'
os.chdir(file_path)
data=pd.read_csv("tsldata.csv")
os.chdir(output_path)
dic_file = r"D:\pythonProject\lda\stop_dic\dict.txt"
stop_file = r"D:\pythonProject\lda\stop_dic\stopwords.txt"

#分词
def chinese_word_cut(mytext):
    jieba.load_userdict(dic_file)  # 加载用户词典
    jieba.initialize()  # 手动初始化（可选）

    # 加载用户停用词表
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")

    stop_list = []  # 存储用户停用词
    flag_list = ['n', 'nz', 'vn']  # 指定在jieba.posseg分词函数中只保存n：名词、nz：其他专名、vn：动名词
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    word_list = []
    seg_list = psg.cut(mytext)  # jieba.posseg分词

    # 原称之为粑粑山型的 词语过滤
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)  # 只匹配所有中文
        find = 0  # 标志位
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:  # 长度小于2或者在用户停用词表中，将被过滤
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:  # 标志位为0且是需要的词性则添加至word_list
            word_list.append(word)
    return (" ").join(word_list)
data["content_cutted"] = data.content.apply(chinese_word_cut)


#LDA分析
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

n_features = 1000 #提取1000个特征词语
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(data.content_cutted)

n_topics = 7 # 手动指定分类数
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                doc_topic_prior=0.1,
                                topic_word_prior=0.01,
                               random_state=0)
lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword

#每个主题输出多少个高频词
n_top_words = 10
tf_feature_names = tf_vectorizer.get_feature_names()
topic_word = print_top_words(lda, tf_feature_names, n_top_words)
