import os
from transformers import BertTokenizer, BertModel

os.environ['TRANSFORMERS_CACHE'] = '/project/tantra/y.zichen/hf_cache'

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=os.environ['TRANSFORMERS_CACHE'])
model = BertModel.from_pretrained(model_name, cache_dir=os.environ['TRANSFORMERS_CACHE'])

text = "This is a sample text."
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.encode(text)
