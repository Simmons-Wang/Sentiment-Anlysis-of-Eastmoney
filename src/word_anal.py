
import jieba
from os import path
import pandas as pd
from tqdm import tqdm
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType
import os

os.chdir("C:\\Users\\Simmons\\PycharmProjects\\数据挖掘大作业")

data1 = pd.read_csv('./data_base/东方财富股吧标题.csv')
text = list(data1.title)
text = ','.join(text)
# tags = jieba.analyse.extract_tags(text,topK=200,withWeight=False)
# text1 = " ".join(tags)

tl = jieba.lcut_for_search(text,HMM=True)
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
count.to_csv('./data_out/词频统计.csv',index=None)
count_tuples = [tuple(xi) for xi in count.values]
del count_tuples[0]
c = (
    WordCloud()
    .add("", count_tuples, word_size_range=[10, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title="东方财富"))
    .render("./fig/东方财富.html")
)
