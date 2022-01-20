import numpy as np
import pandas as pd
from src.Genetic_algorithm import GA
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import os

os.chdir("C:\\Users\\Simmons\\PycharmProjects\\数据挖掘大作业")


class FeatureSelection(object):
    def __init__(self, aLifeCount=10):
        self.columns =  \
            ['target', 'A股', '回升', '湖北', '涨停', '恒生指数', '道琼斯', '平仓', '补', '建仓',
             '量比', '头寸', '跌停', '空头', '港股', '多头', '上涨', '股票', '蓝筹股', '套牢', '上海', '上证',
             '绩优股', '淘宝', '涨停板', '牛股', '升值', '熊市', '成分股', '放量', '股市', '效益', '财产', '期货',
             '妖股', '换手率', '利空', 'K线', '调节', '上证指数', '收盘价', '创业板指数', '概念股', '原油',
             '企业上市', '华南', '多头和空头什么意思', '开盘价', '股份有限公司', '增幅', '恒生', '盈利',
             '损失', '利润率', '熔断', '股票成交量', '证券公司是做什么的', '三板市场', '债券', '高中生炒股赚4.5亿',
             '证券投资基金', '毛利', '补仓', '流通股', '波动', '净利率', '创业板指', '半导体芯片', '休市', '黄金',
             '可转债', '战略管理', '股东', '盘整', '日内交易者', '新基建', '资产', 'ST股', '投资', '量价关系', '持仓',
             '抛售', '成交量', '爆仓', '创业板', '市盈率', '涨停是什么意思', '短线', '期货K线', '做多', '美股熔断',
             '保险', '空仓', 'A股市场', '洗盘', '科技股', '做空', '量比是什么意思', '外围市场', '炒股', '投行',
             '沪深300', '振幅', '石油', '缩量下跌', '空头头寸', '技术分析', '委比', '股票基础', '牛市', '放量上涨',
             '指数基金', '到期日', '收盘时间', '白马股', '崩盘', '反弹', '道琼斯指数', 'B股', 'MACD', '商业银行',
             '科技资讯', '跌停是什么意思', '日经225指数', '股票基础知识入门', '证券基础知识', '重组', '折价率',
             '买空', '香港股市', '营业收入', '大盘', '调研报告', '回落', '证券开户']
        self.train_data = pd.read_excel('./data_base/train_feature.xlsx')
        self.validate_data = pd.read_excel('./data_base/validate_feature.xlsx')
        self.lifeCount = aLifeCount
        self.ga = GA(aCrossRate=0.7,
                     aMutationRage=0.1,
                     aLifeCount=self.lifeCount,
                     aGeneLenght=len(self.columns) - 1,
                     aMatchFun=self.matchFun())

    def mean_error_score(self, order):
        print(order)
        features = self.columns[1:]
        features_name = []
        for index in range(len(order)):
            if order[index] == 1:
                features_name.append(features[index])

        y_train = np.array(self.train_data['target'], dtype=np.float)
        reg = linear_model.LinearRegression()
        reg.fit(self.train_data[features_name], y_train)
        y_test = np.array(self.validate_data['target'], dtype=np.float)
        y_pred = reg.predict(self.validate_data[features_name])
        score = mean_squared_error(y_test, y_pred)
        score = 1 / score
        return score

    def matchFun(self):
        return lambda life: self.mean_error_score(life.gene)

    def run(self, n=0):
        distance_list = [ ]
        generate = [ index for index in range(1, n + 1) ]
        while n > 0:
            self.ga.next()
            # distance = self.auc_score(self.ga.best.gene)
            distance = self.ga.score
            distance_list.append(distance)
            print(("第%d代 : 当前最好特征组合的线下验证结果为：%f") % (self.ga.generation, distance))
            n -= 1

        print('当前最好特征组合:')
        string = []
        flag = 0
        features = self.columns[1:]
        for index in self.ga.gene:
            if index == 1:
                string.append(features[flag])
            flag += 1
        print(string)
        print('线下最高为均方误差的倒数：', self.ga.score)

        '''画图函数'''
        plt.plot(generate, distance_list)
        plt.xlabel('generation')
        plt.ylabel('distance')
        plt.title('generation--mean-error-score')
        plt.show()
        plt.savefig('./fig/遗传算法选择.png')


