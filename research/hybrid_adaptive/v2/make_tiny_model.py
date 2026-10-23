"""make_tiny_model.py - tiny random Qwen2 + the real tokenizer, only for CPU end-to-end smoke tests."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tiny_model")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  tie_word_embeddings=True)
AutoModelForCausalLM.from_config(cfg).save_pretrained(out)
tok.save_pretrained(out)
print("saved", out)
