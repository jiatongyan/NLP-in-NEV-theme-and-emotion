import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

#导入数据
data=pd.read_csv("tsl-主题情感分析.csv",encoding="gbk")

#分好解释变量和被解释变量
X = data.iloc[:,5:11]
y = data.iloc[:,4]
print(y.head())

#分训练集测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#建模预测
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
model_LR= LogisticRegression()
model_LR.fit(X_train,y_train)

y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix)

#============= 变量重要性可视化 =============
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文宋体
plt.rcParams['axes.unicode_minus']=False #显示负号


# 1、简单排序,正负分开按顺序
coef_LR = pd.Series(model_LR.coef_.flatten(),index = X_test.columns,name = 'Var')

plt.figure(figsize=(8,8))
coef_LR.sort_values().plot(kind='barh')
plt.title("各主题重要性",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
