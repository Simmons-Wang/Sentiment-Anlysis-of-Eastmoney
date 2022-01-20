from src import baidu_index, bp, dfcf_title, PCA_keras, Feature_selection_genetic_algorithm
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import jieba
from os import path
import pandas as pd
from tqdm import tqdm
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from pyecharts.charts import HeatMap
warnings.filterwarnings("ignore")

# 由于包含爬虫和参数优化，代码运行时间大约4.5小时
os.chdir("C:\\Users\\Simmons\\PycharmProjects\\DataMining")

# 爬取东方财富标题
Cookie = 'qgqp_b_id=526bd1213504298bebf0750294c2e5c9; HAList=a-sh-601688-%u534E%u6CF0%u8BC1%u5238; em_hq_fls=js; emshistory=%5B%22%E8%8B%8F%E5%AE%81%E6%98%93%E8%B4%AD%22%2C%22%E4%B8%8A%E6%B5%B7%E5%8C%BB%E8%8D%AF%22%2C%22%E5%8D%8E%E6%B3%B0%E8%AF%81%E5%88%B8%22%5D; pitype=web; st_si=21971682780875; st_asi=delete; st_pvi=37151177811914; st_sp=2019-09-15%2011%3A11%3A43; st_inirUrl=https%3A%2F%2Fcn.bing.com%2F; st_sn=6; st_psi=20200508171817899-117001301474-3439656118'
results = []
for i in tqdm(range(10000)):
    url = 'http://guba.eastmoney.com/list,zssh000001_{0}.html'.format(i)
    review_title, review_time = dfcf_title.get_info(url)
    for tit in review_title:
        results.append(tit.text)
    time.sleep(0.5)

data_out = pd.DataFrame(columns=['title'])
data_out['title'] = results
data_out.to_csv('./data_out/东方财富股吧标题.csv', index=None)


# 分词与词频统计
data1 = pd.read_csv('./data_base/东方财富股吧标题.csv')
text = list(data1.title)
text = ','.join(text)
tl = jieba.lcut_for_search(text, HMM=True)
textdf = pd.DataFrame(columns=['text'])
textdf['text'] = tl
count = textdf['text'].value_counts()

stop = open('./data_base/stop_word.txt', 'r+', encoding='utf-8')
stopword = stop.read().split("\n")

wordlist = []
for key in tqdm(tl):
    if key not in stopword:
        if key !=' ':
            wordlist.append(key)

textdf = pd.DataFrame(columns=['text'])
textdf['text'] = wordlist
count = textdf['text'].value_counts()
count = count.reset_index()
count.to_csv('./data_out/词频统计.csv', index=None)
count_tuples = [tuple(xi) for xi in count.values]
del count_tuples[0]
c = (
    WordCloud()
    .add("", count_tuples, word_size_range=[10, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title="东方财富"))
    .render("./fig/东方财富.html")
)

# 关键词词库扩充
data_text = pd.read_csv('./data_out/词频统计.csv')
data_rel = pd.DataFrame(columns=['origin', 'word', 'pv', 'ratio', 'sim'])
for word in tqdm(data_text['index'].head(1000)):
    try:
        a = baidu_index.getMulti(word)
        for item in a:
            item['origin'] = word
            data_rel = data_rel.append(item, ignore_index=True)
    except:
        pass
    time.sleep(2)
data_rel.to_csv('/data_out/关联词.csv', encoding='gbk',index=None)
data_freq = pd.read_csv('./data_out/关联词.csv', encoding='gbk')
freq = data_freq['word'].value_counts()
freq.to_excel('./data_out/关联词词频.xlsx', encoding='gbk')
data_word = pd.read_excel('./data_base/关联词词频.xlsx', encoding='gbk')

count_tuples = [tuple(xi) for xi in data_word.head(140).values]
c = (
    WordCloud()
    .add("", count_tuples, word_size_range=[10, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title="最后关键词词库"))
    .render("./fig/最终关键词词库.html")
)

# 爬取日度搜索指数
word_list = list(data_word['word'])[0:139]

word_dict = {}
for w in tqdm(word_list):
    try:
        a = baidu_index.getIndex(w)
        word_dict[w] = a
    except :
        pass
    time.sleep(3)

word_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in word_dict.items()]))
word_df.to_excel('./data_out/搜索指数.xlsx', encoding='gbk', index=None)

# lasso进行特征选择在Stata中完成


# 遗传算法进行特征选择
fs = Feature_selection_genetic_algorithm.FeatureSelection(aLifeCount=20)
rounds = 100  # 算法迭代次数 #
fs.run(rounds)


# 随机森林进行特征选择
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

# lasso选择出的关键词
word_selected1 = ['跌停', '空头', '港股', '多头', '妖股', '收盘价', '创业板指数', '原油', '开盘价', '休市', '可转债',
                  '盘整', '日内交易者', '短线', '保险', '空仓', '振幅', '石油', '缩量下跌', '委比', '牛市', '崩盘',
                  '道琼斯指数', '商业银行', '跌停是什么意思', '香港股市']

# 遗传算法选择出的关键词
word_selected2 = ['A股', '回升', '涨停', '补', '头寸', '多头', '绩优股', '淘宝', '牛股', '升值', '熊市', '放量',
                  '股市', '效益', '财产', '利空', '调节', '原油', '华南', '多头和空头什么意思', '开盘价', '增幅',
                  '利润率', '证券投资基金', '毛利', '补仓', '流通股', '波动', '净利率', '创业板指', '休市', '黄金',
                  '可转债', '盘整', '投资', '量价关系', '抛售', '成交量', '市盈率', '涨停是什么意思', '期货K线',
                  '美股熔断', '保险', '空仓', 'A股市场', '沪深300', '石油', '技术分析', '委比', '股票基础',
                  '指数基金', '到期日', '白马股', 'MACD', '股票基础知识入门', '折价率', '大盘', '证券开户']

# 随机森林选择出的关键词
word_selected3 = ['A股', '委比', '黄金', '商业银行', '股市', '损失', '华南', '换手率', '资产', '反弹', '蓝筹股',
                  '香港股市', '牛市', '牛股', '道琼斯指数', '战略管理', '财产', '调研报告', '量比是什么意思',
                  '美股熔断', '恒生指数', '可转债', '上海', '道琼斯', '熔断']


df_selected1 = pd.DataFrame({'lasso':word_selected1})
df_selected2 = pd.DataFrame({'遗传算法':word_selected2})
df_selected3 = pd.DataFrame({'随机森林':word_selected3})
df_list = [df_selected1,df_selected2,df_selected3]
df_selected = pd.concat(df_list,axis=1)
df_selected.to_excel('./data_out/选出的重要关键词.xlsx',encoding='gbk',index=False)
data_base.to_csv('./data_base/回归数据.csv', encoding='gbk', index=None)
result.to_csv('./data_base/随机森林选择.csv', encoding='gbk', index=None)

# 主成分法构建情绪指数
data_base2 = pd.read_csv('./data_base/回归数据.csv', encoding='gbk')

word_selected_list = [word_selected1,word_selected2,word_selected3]

price = list(data_base2['price'])
final_score = []
for ws in word_selected_list:

    word_list = ws
    data_base = data_base2[word_list]
    data_base = (data_base - data_base.min()) / (data_base.max() - data_base.min())
    data_tr = data_base.values
    a, b, c = PCA_keras.pcan(data_base, data_tr, 25)
    fin_score = np.dot(a, c.T)
    final_score.append(fin_score)
    y1 = stats.spearmanr(fin_score, price)
    print(y1)
    y2 = stats.pearsonr(fin_score, price)
    print(y2)

# 构建bp神经网络,基于随机森林选择出的关键词
fin_score = final_score[2]
data = pd.DataFrame({'y': price[4:], 'x1': price[3:-1], 'x2': price[2:-2],
                     'x3': price[1:-3], 'x4': price[0:-4], ' x5': fin_score[3: -1], 'x6': fin_score[2:-2]})

# 先划分训练集和测试集，再进行归一化
data_train = data.head(220)
data_test = data.tail(18)
data_train = (data_train - data_train.min()) / (data_train.max() - data_train.min())
data_test = (data_test - data_test.min()) / (data_test.max() - data_test.min())

x_train, y_train, x_test, y_test = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values,\
                                   data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

# 参数优化
alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
nums = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

alphas = [0.0025, 0.005, 0.0075, 0.01]
nums = [10, 11, 12, 13, 14,15]

LOSS = []
for a in tqdm(alphas):
    MSE = []
    for n in tqdm(nums):
        num_epochs = 500
        model = PCA_keras.build_model(n, a, x_train)
        history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=1, verbose=0)
        loss, mse = model.evaluate(x_train, y_train)
        MSE.append(mse)
    LOSS.append(MSE)

loss_df = pd.DataFrame({'0.0025': LOSS[0], '0.005': LOSS[1], '0.0075': LOSS[2],
                        '0.01': LOSS[3]})
loss_df.index = nums
loss_df.to_csv('./data_out/参数优化.csv', encoding='gbk')

value = []
for i in range(0, len(alphas)):
    for j in range(0, len(nums)):
        mse = LOSS[i][j]
        l = [i, j, mse]
        value.append(l)
c = (
    HeatMap()
    .add_xaxis(alphas)
    .add_yaxis("", nums, value)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="参数优化"),
        visualmap_opts=opts.VisualMapOpts(min_=0, max_=0.008),
    )
    .render("./fig/参数优化.html")
)

num_epochs = 5000
model = PCA_keras.build_model(14, 0.005, x_train)
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=1, verbose=0)
predicts = model.predict(x_train)
predictt = model.predict(x_test)

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
plt.savefig('./fig/loss.png')

x = pd.DataFrame({'y':y_train,'p':predicts[:,0]})
test =pd.DataFrame({'y':y_test,'p':predictt[:,0]})

r_s = PCA_keras.r_square(x.y, x.p)
r_t = PCA_keras.r_square(test.y,test.p)

test.plot.line()
plt.show()
plt.savefig('./fig/测试集对比图.png')

x.plot.line()
plt.show()
plt.savefig('./fig/训练集对比图.png')

# 基于numpy搭建bp神经网络
trainX = x_train.T
testX = x_test.T
trainY = y_train.reshape(1, 220)
testY = y_test.reshape(1, 18)

structure = [6, 20, 12, 12, 1]
para = bp.finalModel(trainX, trainY, structure, learningRate=0.001, numIters=10000, pringCost=False)
predTrain, accTrain = bp.predict(trainX, para)
predTest, accTest = bp.predict(testX, para)

pre = pd.DataFrame({'y': y_train, 'p': predTrain[0, :]})
r_p = PCA_keras.r_square(pre.y, pre.p)

pre.plot.line()
plt.show()
plt.savefig('./fig/基于numpy搭建神经网络的训练集对比图.png')

pretest = pd.DataFrame({'y': y_test, 'p': predTest[0, :]})
pretest.plot.line()
plt.show()
plt.savefig('./fig/基于numpy搭建神经网络的测试集对比图.png')