import gensim
from gensim import corpora, models

PATH = "D:/pythonProject/byd-分词.csv"  # 已经进行了分词的文档（如何分词前面的文章有介绍）

file_object2 = open(PATH, encoding='gbk', errors='ignore').read().split('\n')
texts = []  # 建立存储分词的列表
for i in range(len(file_object2)):
    result = []
    seg_list = file_object2[i].split()  # 读取没一行文本
    for w in seg_list:  # 读取每一行分词
        result.append(w)
    texts.append(result)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


def lda_model_values(num_topics, corpus, dictionary):
    x = []  # x轴
    perplexity_values = []  # 困惑度
    coherence_values = []  # 一致性
    model_list = []  # 存储对应主题数量下的lda模型,便于生成可视化网页

    for topic in range(num_topics):
        print("主题数量：", topic + 1)
        lda_model = models.LdaModel(corpus=corpus, num_topics=topic + 1, id2word=dictionary, chunksize=2000, passes=20,
                                    iterations=400)
        model_list.append(lda_model)
        x.append(topic + 1)
        perplexity_values.append(lda_model.log_perplexity(corpus))

        coherencemodel = models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("该主题评价完成\n")
    return model_list, x, perplexity_values, coherence_values

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    from pylab import xticks, yticks, np

    # 调用准备函数
    model_list, x, perplexity_values, coherence_values = lda_model_values(10, corpus, dictionary)

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, perplexity_values, marker="o")
    plt.title("主题建模-困惑度")
    plt.xlabel('主题数目')
    plt.ylabel('困惑度大小')
    xticks(np.linspace(1, 10, 10, endpoint=True))  # 保证x轴刻度为1

    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x, coherence_values, marker="o")
    plt.title("主题建模-一致性")
    plt.xlabel("主题数目")
    plt.ylabel("一致性大小")
    xticks(np.linspace(1, 10, 10, endpoint=True))

    plt.show()







