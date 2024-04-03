# 导入我们所需的库 as：即给库取别名，方便书写
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 定义数据
data=pd.read_csv("散点图.csv")
y=data['tsl情感']
x=data['tsl主题']

# 绘图
# 1. 确定画布
plt.figure(figsize=(8, 4))  # figsize:确定画布大小
# 2. 绘图
plt.scatter(x,  # 横坐标
            y,  # 纵坐标
            c='blue', # 点的颜色
        
)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 3.展示图形
plt.legend()  # 显示图例

plt.show()  # 显示所绘图形
