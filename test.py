# -*-coding = utf-8 -*-
# @Time : 2023/12/30 16:51
# @Author : 万锦
# @File : test.py
# @Softwore : PyCharm


import pandas as pd
df = pd.read_csv("./data/日照数据_test.txt",index_col=0)
print(df.head())
df = pd.read_csv("./data/日照时数_test.txt",index_col=0)
print(df)