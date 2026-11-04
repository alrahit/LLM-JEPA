"""Makes eval_spider.py read the stop tokens from the model instead of hard-coding Qwen's.
Run from the v2 folder:  python patch_eval_eos.py eval_spider.py
Qwen2 behaviour is unchanged (its generation_config lists <|im_end|> and <|endoftext|>)."""
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"
fix = lambda s: s.replace("\n", nl)
if "generation_config.eos_token_id" in text:
    sys.exit("already patched, nothing to do")
old = fix('    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]\n')
new = fix(
'    ge = model.generation_config.eos_token_id\n'
'    eos_ids = {int(x) for x in ([ge] if isinstance(ge, int) else (ge or []))}\n'
'    if tok.eos_token_id is not None:\n'
'        eos_ids.add(int(tok.eos_token_id))\n'
'    eos_ids = sorted(eos_ids)\n'
'    print("stop token ids:", eos_ids, [tok.convert_ids_to_tokens(i) for i in eos_ids], flush=True)\n')
if text.count(old) != 1:
    sys.exit(f"cannot patch: the stop-token line was found {text.count(old)} times (expected 1). Send me the output.")
open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new))
print("patched", path)
