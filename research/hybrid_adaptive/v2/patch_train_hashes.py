"""Adds SHA-256 hashes of the train / validation / test files to every run's args.json.
Run from the v2 folder:  python patch_train_hashes.py train.py
Safe to run once: it refuses to patch a file that is already patched."""
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"
fix = lambda s: s.replace("\n", nl)

if "data_sha256" in text:
    sys.exit("already patched, nothing to do")

helper_anchor = fix("def parse():\n")
helper = fix(
'def sha256_file(path):\n'
'    import hashlib\n'
'    h = hashlib.sha256()\n'
'    with open(path, "rb") as f:\n'
'        for chunk in iter(lambda: f.read(1 << 20), b""):\n'
'            h.update(chunk)\n'
'    return h.hexdigest()\n'
'\n\n')

arg_anchor = fix('    ap.add_argument("--save_every_epoch", action="store_true")\n')
arg_new = arg_anchor + fix(
'    ap.add_argument("--val_file", default="", help="only hashed and recorded, never read for training")\n'
'    ap.add_argument("--test_file", default="", help="only hashed and recorded, never read for training")\n')

cfg_anchor = fix('    with open(os.path.join(args.out_dir, "args.json"), "w") as f:\n        json.dump(vars(args), f, indent=1)\n')
cfg_new = fix(
'    cfg = vars(args)\n'
'    cfg["data_sha256"] = {k: sha256_file(cfg[k]) for k in ("train_file", "val_file", "test_file") if cfg.get(k)}\n'
'    with open(os.path.join(args.out_dir, "args.json"), "w") as f:\n        json.dump(cfg, f, indent=1)\n')

for name, a in (("def parse()", helper_anchor), ("--save_every_epoch", arg_anchor), ("args.json block", cfg_anchor)):
    if text.count(a) != 1:
        sys.exit(f"cannot patch: anchor '{name}' found {text.count(a)} times (expected 1). Send me the output.")
text = text.replace(helper_anchor, helper + helper_anchor).replace(arg_anchor, arg_new).replace(cfg_anchor, cfg_new)
open(path, "w", encoding="utf-8", newline="").write(text)
print("patched", path)
