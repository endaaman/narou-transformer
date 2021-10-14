import math
import time
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from endaaman import TorchCommander
from datasets import get_wiki_data, get_translate_data, get_collate_fn


class BertForSequenceClassification_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()

        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class Trainer(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('-e', '--epoch', type=int, default=18)
        parser.add_argument('-b', '--batch-size', default=2)

    def run_translate(self, args):
        (en_vocab, de_vocab), train_data, val_data, test_data = get_translate_data()
        print('data loaded')

        torch.manual_seed(0)
        SRC_VOCAB_SIZE = len(en_vocab) + 100
        TGT_VOCAB_SIZE = len(de_vocab)
        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3

        model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                         NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)


        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        model = model.to(self.device)

        generate_batch_fn = get_collate_fn(en_vocab, de_vocab)
        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=generate_batch_fn)
        valid_loader = DataLoader(val_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=generate_batch_fn)
        test_iter = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=generate_batch_fn)

        PAD_IDX = en_vocab.stoi['<pad>']
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        for epoch in range(1, self.args.epoch + 1):
            start_time = timer()
            train_loss = self.train_epoch(model, optimizer, train_loader)
            end_time = timer()
            val_loss = self.evaluate(model, valid_loader)
            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")

    def train_epoch(self, model, optimizer, loader):
        model.train()
        losses = 0
        for src, tgt in loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            print(src)
            print(tgt)
            print(src.size())
            print(tgt.size())
            # tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

            print('src', src.size(), 'src mask', src_mask.size(), 'tgt', tgt.size(), 'tgt mast', tgt_mask.size())

            logits = model(
                src,
                tgt,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask)

            optimizer.zero_grad()
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            losses += loss.item()

        return losses / len(loader)


    def evaluate(self, model, loader):
        model.eval()
        losses = 0

        for src, tgt in loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(loader)


Trainer().run()
