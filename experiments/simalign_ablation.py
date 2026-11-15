# -*- coding: utf-8 -*-
"""SimAlign part of the CombAlign ablation on the en-is gold set (dev 220, test 384): model × level × method × κ,
plus votes over the SimAlign variants. CPU is enough (604 sentences). Writes results/simalign_{dev,test}.tsv and
results/predictions/*.pharaoh for the later ensembles with Giza++/fast_align/eflomal."""
import itertools, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core, simalign as sa

R = f"{ROOT}/results"
os.makedirs(f"{R}/predictions", exist_ok=True)
for split in ("dev", "test"):
    src, trg, gold = core.read_gold(f"{ROOT}/data/gold/{split}")
    rows, preds = [], {}
    for model in ("mbert", "xlmr"):
        E = sa.Embedder(model, device="cpu"); cache = {}
        for level, method, kappa in itertools.product(("word", "bpe"), ("argmax", "itermax", "match"), (0.0, 0.09)):
            name = f"{model}.{level}.{method}.k{kappa}"
            p = sa.align_corpus(src, trg, model, method, level, kappa, embedder=E, cache=cache)
            preds[name] = p
            core.write_pharaoh(f"{R}/predictions/{split}.{name}.pharaoh", p)
            rows.append((name, core.evaluate(p, gold))); print(split, *rows[-1], flush=True)
    names = list(preds)
    for k in (2, 3, 4, 6, 8, 12):
        rows.append((f"vote{k}/{len(names)}.all", core.evaluate(core.vote([preds[n] for n in names], k), gold)))
    for m in ("mbert", "xlmr"):
        sub = [n for n in names if n.startswith(m)]
        for k in (2, 4, 6):
            rows.append((f"vote{k}/{len(sub)}.{m}", core.evaluate(core.vote([preds[n] for n in sub], k), gold)))
    with open(f"{R}/simalign_{split}.tsv", "w") as f:
        f.write("system\tP\tR\tF1\tAER\tedges\n")
        for n, r in sorted(rows, key=lambda x: -x[1]["F1"]):
            f.write(f"{n}\t{r['P']}\t{r['R']}\t{r['F1']}\t{r['AER']}\t{r['edges']}\n")
