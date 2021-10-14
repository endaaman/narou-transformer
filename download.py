import requests
import pandas as pd
import json
import time
import datetime
import gzip
from tqdm import tqdm
import xlsxwriter
import sqlite3

from endaaman import Commander

class Downloader(Commander):

    def arg_all(self, parser):
        parser.add_argument('-o', '--out', choices=['csv', 'excel', 'sql'], default='csv')
        parser.add_argument('-a', '--adult', action='store_true')

    def run_all(self):
        print('started ', datetime.datetime.now())
        tqdm.pandas()
        interval = 2

        now_day = datetime.datetime.now().strftime('%Y_%m_%d')
        if self.args.adult:
            api_url = 'https://api.syosetu.com/novelapi/api/'
        else:
            api_url = 'https://api.syosetu.com/novel18api/api/'

        ext = {
            'csv': '.csv',
            'excel': '.xlsx',
            'sql': '.sqlite',
        }[self.args.out]
        filename = f'data/Narou_All_OUTPUT_{now_day}{ext}'

        df = pd.DataFrame()
        payload = {'out': 'json', 'gzip':5, 'of':'n', 'lim':1}
        res = requests.get(api_url, params=payload).content
        r =  gzip.decompress(res).decode('utf-8')
        allcount = json.loads(r)[0]['allcount']

        print(f'target count: {allcount}')

        all_queue_cnt = (allcount // 500) + 10

        nowtime = datetime.datetime.now().timestamp()
        lastup = int(nowtime)
        for i in tqdm(range(all_queue_cnt)):
            payload = {'out': 'json','gzip':5,'opt':'weekly','lim':500,'lastup':'1073779200-'+str(lastup)}
            # なろうAPIにリクエスト
            cnt = 0
            while cnt < 5:
                try:
                    res = requests.get(api_url, params=payload, timeout=30).content
                    break
                except:
                    print('Connection Error')
                    cnt = cnt + 1
                    time.sleep(120) #接続エラーの場合、120秒後に再リクエストする

            r =  gzip.decompress(res).decode('utf-8')
            # pandasのデータフレームに追加する処理
            df_temp = pd.read_json(r)
            df_temp = df_temp.drop(0)
            df = pd.concat([df, df_temp])
            last_general_lastup = df.iloc[-1]['general_lastup']
            lastup = datetime.datetime.strptime(last_general_lastup, '%Y-%m-%d %H:%M:%S').timestamp()
            lastup = int(lastup)
            time.sleep(interval)

        df = df.drop('allcount', axis=1)
        df.drop_duplicates(subset='ncode', inplace=True)
        df = df.reset_index(drop=True)
        print('done fetching', datetime.datetime.now())
        print('start export')

        if self.args.out == 'sql':
            conn = sqlite3.connect(filename)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            df.to_sql('novel_data', conn, if_exists='replace')
            c.close()
            conn.close()
        elif self.args.out == 'excel':
            writer = pd.ExcelWriter(filename, options={'strings_to_urls': False}, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Sheet1')
            writer.close()
        elif self.args.out == 'csv':
            df.to_csv(filename)

        print('done:', datetime.datetime.now())



Downloader().run()
