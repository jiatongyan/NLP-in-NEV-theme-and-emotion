#导入模块
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import re
import jieba
import codecs

#导入数据
data=pd.read_csv("byddata.csv")

# 评论情感分析
from snownlp import SnowNLP
#f = open('biyadidata.csv',encoding='gbk')
#line = f.readline()
with open('stop_word/HGD_StopWords.txt','r',encoding='utf-8') as f:
    stopwords=set([line.replace('\n','')for line in f])
f.close()
sum=0
count=0
for i in range(len(data['content'])):
    line=jieba.cut(data.loc[i,'content'])           #分词
    words=''
    for seg in line:
        if seg not in stopwords and seg!=" ":        #文本清洗
            words=words+seg+' '
    if len(words)!=0:
        print(words)        #输出每一段评论的情感得分
        d=SnowNLP(words)
        print('{}'.format(d.sentiments))
        data.loc[i,'sentiment_score']=float(d.sentiments)     #原数据框中增加情感得分列
        sum+=d.sentiments
        count+=1
score=sum/count
print('finalscore={}'.format(score))    #输出最终情感得分

#将情感得分结果保存为新的csv文件
data.to_csv('result.csv',encoding='gbk',header=True)