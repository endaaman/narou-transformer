import io
import torch

from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


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
    return train_data, val_data, test_data
