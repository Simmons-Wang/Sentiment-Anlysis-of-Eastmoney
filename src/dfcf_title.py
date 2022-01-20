import requests
from bs4 import BeautifulSoup
import os

os.chdir("C:\\Users\\Simmons\\PycharmProjects\\数据挖掘大作业")

Cookie = 'qgqp_b_id=526bd1213504298bebf0750294c2e5c9; HAList=a-sh-601688-%u534E%u6CF0%u8BC1%u5238; em_hq_fls=js; emshistory=%5B%22%E8%8B%8F%E5%AE%81%E6%98%93%E8%B4%AD%22%2C%22%E4%B8%8A%E6%B5%B7%E5%8C%BB%E8%8D%AF%22%2C%22%E5%8D%8E%E6%B3%B0%E8%AF%81%E5%88%B8%22%5D; pitype=web; st_si=21971682780875; st_asi=delete; st_pvi=37151177811914; st_sp=2019-09-15%2011%3A11%3A43; st_inirUrl=https%3A%2F%2Fcn.bing.com%2F; st_sn=6; st_psi=20200508171817899-117001301474-3439656118'


def get_info(u):
    """
    :param : 获取标题数据的url
    :return: 获得的标题数据和发布时间
    """
    url = u
    # 设置requests请求的 headers
    headers = {
        'User-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36",
        # 设置get请求的User-Agent，用于伪装浏览器UA
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Cookie': Cookie,
        'Connection': 'keep-alive',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Host': 'guba.eastmoney.com',
    }
    req = requests.get(url, headers=headers)
    str_data = req.content
    ns = BeautifulSoup(str_data, 'html.parser')
    title = ns.find_all("span", attrs={"class": "l3 a3"})
    mtime = ns.find_all("span", attrs={"class": "l5 a5"})
    return title, mtime


