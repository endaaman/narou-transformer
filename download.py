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

    def run_all(self, args):
        tqdm.pandas()
        interval = 2
        is_narou = True

        now_day = datetime.datetime.now()
        now_day = now_day.strftime("%Y_%m_%d")
        if is_narou:
            filename = f'data/Narou_All_OUTPUT_{now_day}.xlsx'
            sql_filename = f'data/Narou_All_OUTPUT_{now_day}.sqlite3'
            api_url = 'https://api.syosetu.com/novelapi/api/'
        else:
            filename = f'data/Narou_18_ALL_OUTPUT_{now_day}.xlsx'
            sql_filename = f'data/Narou_18_ALL_OUTPUT_{now_day}.sqlite3'
            api_url = 'https://api.syosetu.com/novel18api/api/'

        is_save_sqlite = False

        df = pd.DataFrame()
        payload = {'out': 'json', 'gzip':5, 'of':'n', 'lim':1}
        res = requests.get(api_url, params=payload).content
        r =  gzip.decompress(res).decode("utf-8")
        allcount = json.loads(r)[0]['allcount']

        print(f'対象作品数  {allcount}');

        all_queue_cnt = (allcount // 500) + 10

        nowtime = datetime.datetime.now().timestamp()
        lastup = int(nowtime)
        for i in tqdm(range(all_queue_cnt)):
            payload = {'out': 'json','gzip':5,'opt':'weekly','lim':500,'lastup':"1073779200-"+str(lastup)}
            # なろうAPIにリクエスト
            cnt=0
            while cnt < 5:
                try:
                    res = requests.get(api_url, params=payload, timeout=30).content
                    break
                except:
                    print("Connection Error")
                    cnt = cnt + 1
                    time.sleep(120) #接続エラーの場合、120秒後に再リクエストする

            r =  gzip.decompress(res).decode("utf-8")
            # pandasのデータフレームに追加する処理
            df_temp = pd.read_json(r)
            df_temp = df_temp.drop(0)
            df = pd.concat([df, df_temp])
            last_general_lastup = df.iloc[-1]["general_lastup"]
            lastup = datetime.datetime.strptime(last_general_lastup, "%Y-%m-%d %H:%M:%S").timestamp()
            lastup = int(lastup)
            time.sleep(interval)

        df = df.drop("allcount", axis=1)
        df.drop_duplicates(subset='ncode', inplace=True)
        df = df.reset_index(drop=True)
        print("export_start",datetime.datetime.now())
        try:
            writer = pd.ExcelWriter(filename, options={'strings_to_urls': False}, engine='xlsxwriter')
            df.to_excel(writer, sheet_name="Sheet1")
            writer.close()
            print('取得成功数  ',len(df));
        except:
            pass

        if is_save_sqlite == True or len(df) >= 1048576:
            conn = sqlite3.connect(sql_filename)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            df.to_sql('novel_data', conn, if_exists='replace')
            c.close()
            conn.close()
            print('Sqlite3形式でデータを保存しました')

        print("start",datetime.datetime.now())
        get_all_novel_info()
        print("end",datetime.datetime.now())



Downloader().run()
