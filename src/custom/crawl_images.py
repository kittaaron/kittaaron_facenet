# 根据文件starName.txt中的明星名字进行下载图片
import os
from icrawler.builtin import BingImageCrawler

path = 'D:\\scrawl_images\\images2\\'   #图片存储路径

file = open('E:\\scrawl_images\\starName.txt', 'r', encoding='utf-8')
lines = file.readlines()
for i, line in enumerate(lines):
    starname = line.strip('\n')
    starname_dir = os.path.join(path, starname)
    if not os.path.exists(starname_dir):
        os.makedirs(starname_dir)
    storage = {'root_dir': starname_dir}
    # 指定存储路径，4个下载线程
    bing_crawler = BingImageCrawler(parser_threads=2, downloader_threads=4, storage=storage)
    # 每个人下载10张图片
    bing_crawler.crawl(keyword=starname, max_num=25)
    print('第{}位明星：{}'.format(i, starname))