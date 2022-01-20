import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

os.chdir("C:\\Users\\Simmons\\PycharmProjects\\数据挖掘大作业")
data_index = pd.read_excel('./data_base/搜索指数.xlsx', encoding='gbk')
data_stock = pd.read_excel('./data_base/上证综指.xlsx', encoding='gbk')

tm = pd.date_range('20190601', periods=366, freq='D')
data_index['time'] = tm
data_base = data_stock.merge(data_index, on='time')

data_base.isnull().sum().sum()
data_base = data_base.fillna(0)
x, y = data_base.iloc[:, 2:].values, data_base.iloc[:, 1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
feat_labels = data_base.columns[2:]
forest = RandomForestRegressor(n_estimators=1000)
forest.fit(x_train, y_train)

importance = forest.feature_importances_
x_selected = x_train[:, importance > 0.01]
ser = pd.Series(importance.tolist())
ser.index = feat_labels
result = ser.sort_values()
result = result.reset_index()

data_base.to_csv('./data_base/回归数据.csv', encoding='gbk', index=None)
result.to_csv('./data_base/随机森林选择.csv', encoding='gbk',index=None)
