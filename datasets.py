import io
from typing import Iterable, List
from collections import Counter

from tqdm import tqdm
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


FILENAME = 'data/Narou_All_OUTPUT_2021_06_17.csv'

genre_id_to_str = {
    101: '異世界',
    102: '現実世界',
    201: 'ハイファンタジー',
    202: 'ローファンタジー',
    301: '純文学',
    302: 'ヒューマンドラマ',
    303: '歴史',
    304: '推理',
    305: 'ホラー',
    306: 'アクション',
    307: 'コメディー',
    401: 'VRゲーム',
    402: '宇宙',
    403: '空想科学',
    404: 'パニック',
    9901: '童話',
    9902: '詩',
    9903: 'エッセイ',
    9904: 'リプレイ',
    9999: 'その他',
    9801: 'ノンジャンル',
}

class NarouDataset(Dataset):
    def __init__(self):
        # df = pd.read_csv(FILENAME)
        # df = dd.read_csv(FILENAME).compute()

        self.df = self.load_csv(FILENAME).dropna(subset=['title'])
        print(self.df)

    def load_csv(self, filename, chunksize=10000):
        chunks = []
        with tqdm(total=817318) as t:
            for chunk in pd.read_csv(FILENAME, chunksize=chunksize):
                t.update(chunksize)
                chunks.append(chunk)
        return pd.concat(chunks)

    def __get__(self, idx):
        pass

    def __getitem__(self, idx):
        pass

    def to_file(self, col='title', path='data/titles.txt'):
        titles = self.df[col].values
        with open(path, mode='w') as f:
            f.writelines([str(t) + '\n' for t in titles])

    def to_gpt2_dataset(self, tokenizer):
        mask = np.random.rand(len(self.df)) < 0.8
        df_train = self.df[mask]
        df_val = self.df[~mask]
        self.to_gpt2_dataset_by_target(tokenizer, df_train, 'train')
        self.to_gpt2_dataset_by_target(tokenizer, df_val, 'val')

    def to_gpt2_dataset_by_target(self, tokenizer, df, target='train'):
        if target not in ['train', 'val']:
            raise ValueError(f'Invalid target: {target}')
        with open(f'data/gpt2_{target}.txt', 'w') as f:
            for row in tqdm(df.itertuples(), total=df.shape[0]):
                genre = genre_id_to_str[row.genre]
                title = str(row.title)
                story = row.story
                tokens = tokenizer.tokenize(story)[:256]
                story = "".join(tokens).replace('▁', '')
                # text = '<s>' + genre + '[SEP]' + story + '[SEP]' + story + '</s>'
                text = '<s>{genre}[SEP]{title}</s>'
                f.write(text + '\n')

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('rinna/japanese-gpt2-medium')
    tokenizer.do_lower_case = True
    ds = NarouDataset()
    # ds.to_gpt2_dataset(tokenizer)
