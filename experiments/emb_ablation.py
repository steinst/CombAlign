# -*- coding: utf-8 -*-
"""Embedding aligners beyond the paper: LaBSE (AccAlign's base, layers 6/8/12), NLLB-200 encoder
(TransAlign; layers 12/8), XLM-R large, multilingual-e5; plus awesome-align's softmax-threshold extraction for
every model (incl. mBERT = awesome-align without fine-tuning). Appends to results/predictions; rows → results/emb_{split}.tsv.
    python experiments/emb_ablation.py [models, default all new + softmax for mbert/xlmr]"""
import itertools, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core, simalign as sa

R = f"{ROOT}/results"
models = (sys.argv[1] if len(sys.argv) > 1 else "labse,labse8,labse12,nllb,nllb8,xlmrL,e5,mbert,xlmr").split(",")
old = {"mbert", "xlmr"}            # already in simalign_ablation: only add the softmax method
for split in ("dev", "test"):
    src, trg, gold = core.read_gold(f"{ROOT}/data/gold/{split}")
    rows = []
    for model in models:
        E = sa.Embedder(model, device="cpu"); cache = {}
        combos = [("bpe", "softmax", 0.0)] if model in old else \
            list(itertools.product(("word", "bpe"), ("argmax", "itermax"), (0.0, 0.09))) + [("bpe", "softmax", 0.0)]
        for level, method, kappa in combos:
            name = f"{model}.{level}.{method}.k{kappa}"
            p = sa.align_corpus(src, trg, model, method, level, kappa, embedder=E, cache=cache)
            core.write_pharaoh(f"{R}/predictions/{split}.{name}.pharaoh", p)
            rows.append((name, core.evaluate(p, gold))); print(split, *rows[-1], flush=True)
        del E, cache
    with open(f"{R}/emb_{split}.tsv", "w") as f:
        f.write("system\tP\tR\tF1\tAER\tedges\n")
        for n, r in sorted(rows, key=lambda x: -x[1]["F1"]):
            f.write(f"{n}\t{r['P']}\t{r['R']}\t{r['F1']}\t{r['AER']}\t{r['edges']}\n")
