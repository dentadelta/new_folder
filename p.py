import torch
print(torch.cuda.is_available())
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

#Time to redownload my games:
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

from transformers import GPT2Tokenizer, OPTModel
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
model = OPTModel.from_pretrained("facebook/opt-350m")

from transformers import T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")

from transformers import BloomTokenizerFast, BloomModel
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b3")
model = BloomModel.from_pretrained("bigscience/bloom-1b3")

from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large")