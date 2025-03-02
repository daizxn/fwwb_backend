# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",cache_dir="cache")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased",cache_dir="cache")