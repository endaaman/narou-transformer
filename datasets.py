import io
from typing import Iterable, List
from collections import Counter

import numpy as np
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import Vocab, build_vocab_from_iterator
# from torchtext.datasets import Multi30k


def get_wiki_data():
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding="utf8"))))

    def data_process(raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(iter(io.open(train_filepath, encoding='utf8')))
    val_data = data_process(iter(io.open(valid_filepath, encoding='utf8')))
    test_data = data_process(iter(io.open(test_filepath, encoding='utf8')))
    return vocab, train_data, val_data, test_data


def get_translate_data():
    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.de.gz', 'train.en.gz')
    val_urls = ('val.de.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

    def build_vocab(filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    def data_process(filepaths):
        raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
            de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long)
            en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)
            data.append((de_tensor_, en_tensor_))
        return data

    train_data = data_process(train_filepaths)
    val_data = data_process(val_filepaths)
    test_data = data_process(test_filepaths)
    return (en_vocab, de_vocab), train_data, val_data, test_data


def get_collate_fn(en_vocab, de_vocab):
    PAD_IDX = de_vocab['<pad>']
    BOS_IDX = de_vocab['<bos>']
    EOS_IDX = de_vocab['<eos>']
    def generate_batch(data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return de_batch, en_batch
    return generate_batch

    # train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
    #                         shuffle=True, collate_fn=generate_batch)
    # valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
    #                         shuffle=True, collate_fn=generate_batch)
    # test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
    #                        shuffle=True, collate_fn=generate_batch)
if __name__ == '__main__':
    (en_vocab, de_vocab), train_data, val_data, test_data = get_translate_data()
    fn = get_collate_fn(en_vocab, de_vocab)
    train_iter = DataLoader(train_data, batch_size=8, collate_fn=fn)
