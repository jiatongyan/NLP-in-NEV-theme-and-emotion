import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("xiang.csv")
box_1, box_2= data['tsl'], data['byd']
box_1=box_1.dropna(how ='all')
box_2=box_2.dropna(how ='all')

plt.figure(figsize=(10, 5))  # 设置画布的尺寸

# boxprops：color箱体边框色，facecolor箱体填充色；
plt.boxplot([box_1, box_2],
            patch_artist=True,
            widths=0.6,
            boxprops={'color': 'blue', 'facecolor': 'pink'})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()  # 显示图像
