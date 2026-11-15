# -*- coding: utf-8 -*-
"""Evaluate an awesome-align fine-tuned model (work/awesome/<name>, see finetune_awesome.sh) on the gold set two
ways: awesome-align's own extraction (softmax, threshold as trained) and our ArgMax/IterMax (simalign.py) → predictions/.
    python experiments/ft_eval.py labse_ft|mbert_ft"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core, simalign as sa
name = sys.argv[1]
layer, thr = (6, "0.1") if name.startswith("labse") else (8, "0.001")
R = f"{ROOT}/results"
E = sa.Embedder(f"{ROOT}/work/awesome/{name}", device="cpu", layer=layer)
for split in ("dev", "test"):
    src, trg, gold = core.read_gold(f"{ROOT}/data/gold/{split}")
    out = f"{ROOT}/work/awesome/{name}.{split}.out"
    subprocess.run(["awesome-align", "--no_cuda", "--model_name_or_path", f"{ROOT}/work/awesome/{name}",
                    "--data_file", f"{ROOT}/work/awesome/{split}.txt", "--output_file", out, "--align_layer", str(layer),
                    "--extraction", "softmax", "--softmax_threshold", thr, "--batch_size", "32"], check=True)
    preds = {f"{name}.awesome": [core.parse_pharaoh(l)[0] for l in open(out)]}
    cache = {}
    for level, method in (("word", "argmax"), ("word", "itermax"), ("bpe", "argmax")):
        preds[f"{name}.{level}.{method}.k0.0"] = sa.align_corpus(src, trg, name, method, level, 0.0, embedder=E, cache=cache)
    with open(f"{R}/ft_{name}_{split}.tsv", "w") as f:
        for n, p in preds.items():
            core.write_pharaoh(f"{R}/predictions/{split}.{n}.pharaoh", p)
            e = core.evaluate(p, gold)
            f.write(f"{n}\t{e['P']}\t{e['R']}\t{e['F1']}\t{e['AER']}\n"); print(split, n, e, flush=True)
