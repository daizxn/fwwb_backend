# Load model directly
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",cache_dir="cache")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased",cache_dir="cache")