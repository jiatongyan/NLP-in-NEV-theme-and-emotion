import pandas as pd
import jieba


#导入数据
data=pd.read_csv("byddata.csv")

with open('stop_word/HGD_StopWords.txt','r',encoding='utf-8') as f:
    stopwords=set([line.replace('\n','')for line in f])
f.close()

for i in range(len(data['content'])):
    line=jieba.cut(data.loc[i,'content'])           #分词
    words=''
    for seg in line:
        if seg not in stopwords and seg!=" ":        #文本清洗
            words=words+seg+' '
    if len(words)!=0:
        print(words)
        data.loc[i, 'sentiment_score'] = words
data.to_csv('fenciresult.csv',encoding='gbk',header=True)
