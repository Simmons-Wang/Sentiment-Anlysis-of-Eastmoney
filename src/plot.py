import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

os.chdir("C:\\Users\\Simmons\\PycharmProjects\\数据挖掘大作业")
data_index = pd.read_excel('./data_base/搜索指数.xlsx', encoding='gbk')
data_index.iloc[:, 0:5].plot.line()
plt.show()
plt.savefig('./fig/部分关键词搜索指数.png')