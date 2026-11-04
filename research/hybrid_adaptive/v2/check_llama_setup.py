import datetime
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
path = snapshot_download(MODEL)  # about 2.5 GB; needs the license accepted and a login
print("downloaded to:", path)
print("revision hash:", os.path.basename(path))

tok = AutoTokenizer.from_pretrained(MODEL)
tpl = tok.chat_template or ""
print("template mentions strftime_now:", "strftime_now" in tpl, "| mentions 'Today Date':", "Today Date" in tpl)

msgs = [{"role": "system", "content": "Convert natural language to SQL."},
        {"role": "user", "content": "For db_id:[x]\n\nHow many rows?"}]
default = tok.apply_chat_template(msgs, tokenize=False)
fixed = tok.apply_chat_template(msgs, tokenize=False, date_string="01 Jan 2000")
today = datetime.date.today().strftime("%d %b %Y")
print("rendered prompt depends on date_string:", default != fixed)
print("today's date appears in the default prompt:", today in default)
print("default prompt:", repr(default))
