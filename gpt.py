import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

from endaaman import TorchCommander
from datasets import genre_id_to_str

class GPT(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('-m', '--model-name')

    def pre_common(self):
        if self.args.model_name:
            model_name = self.args.model_name
        else:
            model_name = 'rinna/japanese-gpt2-medium'
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def arg_predict(self, parser):
        parser.add_argument('-c', '--category', default='異世界')
        parser.add_argument('-l', '--len', type=int, default=10)

    def run_predict(self):
        self.model.eval()
        self.model.to(self.device)

        category_ids = [self.tokenizer.encode(c)[1:-1] for c in genre_id_to_str.values()]
        category_ids = [c for c in category_ids if len(c)>0]
        # print(category_ids)
        # return

        if self.args.category not in genre_id_to_str.values():
            raise ValueError(f'Invalid category: {self.args.category}')
        input_text = f'<s>{self.args.category}[SEP]俺'
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        outpuy_ids = self.model.generate(
            input_ids,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            num_return_sequences=self.args.len,
            max_length=256,
            # bad_words_ids=[[1], [5]] + [[12558], [9105], [931], [521], [8877], [3130], [3840], [22959], [1000], [3204], [3384], [3565], [9639]] ,
            # bad_words_ids=[[1], [5]] + category_ids,
            bad_words_ids=[[1], [5]],
        )
        for sent in self.tokenizer.batch_decode(outpuy_ids):
            sent = sent.split('[SEP]')[1]
            sent = sent.replace('</s>', '')
            print(sent)

GPT().run()
