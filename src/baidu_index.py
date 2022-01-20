import requests
import pandas as pd
import time
from tqdm import tqdm
import os

os.chdir("C:\\Users\\Simmons\\PycharmProjects\\数据挖掘大作业")


def getIndex(word):
    """
        搜索指数
        :param word:所搜索的关键词
        :return:
        """
    url = f"http://index.baidu.com/api/SearchApi/index?word=[[%7B%22name%22:\"{word}\",%22wordType%22:1%7D]]&area=0&startDate=2019-06-01&endDate=2020-05-31"
    rep_json = get_rep_json(url)
    generalRatio = rep_json['data']['generalRatio']
    uniqid = rep_json['data']['uniqid']
    all_index_e = rep_json['data']['userIndexes'][0]['all']['data']
    # pc_index_e = rep_json['data']['userIndexes'][0]['pc']['data']
    # wise_index_e = rep_json['data']['userIndexes'][0]['wise']['data']
    t = getPtbk(uniqid)
    return decrypt_py(t, all_index_e)


def getFeedIndex(word):
    """
    :param word: 关键词
    :return: 资讯指数
    """
    url = f"http://index.baidu.com/api/FeedSearchApi/getFeedIndex?word=[[%7B%22name%22:\"{word}\",%22wordType%22:1%7D]]&area=0&startDate=2019-06-01&endDate=2020-05-31"
    feed_index_data = get_rep_json(url)
    uniqid = feed_index_data['data']['uniqid']
    data = feed_index_data["data"]['index'][0]
    generalRatio = data['generalRatio']  # 资讯指数概览
    e = data['data']
    t = getPtbk(uniqid)

    return decrypt_py(t, e)

def getMulti(word):
    """
    :param word: 搜索的关键词
    :return: 需求图谱中的pv搜索热度；ratio搜索变化率；sim相关性
    """
    url = f"http://index.baidu.com/api/WordGraph/multi?wordlist%5B%5D={word}"
    word_data = get_rep_json(url)['data']['wordlist'][0]
    # if word_data['keyword']:
    #     print(word_data['wordGraph'])
    return word_data['wordGraph']


def getPtbk(uniqid):
    """
    :param uniqid: 指数数据网址中指向密钥的uniqid
    :return: key
    """
    url = f"http://index.baidu.com/Interface/ptbk?uniqid={uniqid}"
    return get_rep_json(url)['data']


def decrypt_py(t, e):
    """
    :param t: 解密的key
    :param e: 加密的数据
    :return: 解析出来的数据
    """
    a = dict()
    length = int(len(t) / 2)
    for o in range(length):
        a[t[o]] = t[length + o]
    r = "".join([a[each] for each in e]).split(",")
    # print(r)

    return r


def get_rep_json(url):
    """
    获取json
    :param url: 请求接口
    :return: json数据
    """
    hearder = {
        "Cookie": 'BIDUPSID=E43159961D154B035826914AF768C291; PSTM=1568907172; BAIDUID=808337D79D013B87CED62FC179F6182E:FG=1; BDUSS=kxOFhPeFp2WmxIOTk4WUZtd1VaRGF2WXB5QWtBelJQb3hwYzlNdlljZnFPOFZlSVFBQUFBJCQAAAAAAAAAAAEAAAD8Pe7DzsvOy87L1qjWqNaoMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOqunV7qrp1eOU; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; H_PS_PSSID=31728_1446_31670_21104_31593_31271_31463_30824_26350; delPer=0; PSINO=3; Hm_lvt_d101ea4d2a5c67dab98251f0b5de24dc=1590242904,1590314012,1590315854,1590396278; bdindexid=f2jto5lpot2icd77mhp5j338k3; RT="z=1&dm=baidu.com&si=b2of2eeh1is&ss=kam8vp78&sl=3&tt=8ag&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf"; Hm_lpvt_d101ea4d2a5c67dab98251f0b5de24dc=1590396412',  # 请填写游览器中的cookie
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36"
    }
    response = requests.get(url, headers=hearder)
    response_data = response.json()
    # print(response_data)
    return response_data


