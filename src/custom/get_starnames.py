# An highlighted block
import json
import requests


def getStarNames(page_num):
    params = []
    for i in range(0, 12 * page_num + 12, 12):
        params.append({'resource_id': 28266, 'query': '明星', 'stat1': '韩国', 'pn': i, 'rn': 12})
    url = 'https://sp0.baidu.com/8aQDcjqpAAV3otqbppnN2DJv/api.php'
    x = 0
    f = open('E:\\scrawl_images\\starName2.txt', 'a', encoding='utf-8')
    for param in params:
        try:
            # http request 爬取人物名称
            res = requests.get(url, params=param)
            js = json.loads(res.text)
            results = js.get('data')[0].get('result')
        except AttributeError as e:
            print(e)
            continue
        for result in results:
            f.write(result['ename'] + '\n')
        if x % 10 == 0:
            print('第%d页......' % x)
        x += 1
    f.close()


if __name__ == '__main__':
    getStarNames(50)