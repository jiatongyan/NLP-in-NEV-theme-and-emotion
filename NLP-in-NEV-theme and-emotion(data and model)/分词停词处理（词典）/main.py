import pandas as pd

df = pd.read_csv("比亚迪评论1.txt",delimiter="\t")

df.to_csv("比亚迪评论1.csv", encoding='utf-8', index=False)