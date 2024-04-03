import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("散点图.csv")
x = data.iloc[:,0:2]
print(x.head())

d = x.corr()

plt.subplots(figsize = (12,12))
sns.heatmap(d,annot = True,vmax = 1,square = True,cmap = "Reds")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

