# -*- coding: utf-8 -*-
"""CombAlign ablation, part 2: statistical aligners × symmetrisation × training size/corpus, then ensembles.
  align   python experiments/ensemble_ablation.py align <aligner> <sizes> [corpus=parice]
          trains on the first N pairs of data/train/<corpus>.{en,is} + dev + test sentences →
          results/predictions/{split}.{aligner}.{corpus}{size}.{sym}.pharaoh and results/stat/{aligner}.{corpus}{size}.tsv
  combine python experiments/ensemble_ablation.py combine
          scores every prediction file + ensembles (EnsSmall 2/3, EnsLarge 3/5 as in the paper, votes over the best
          members) → results/ensemble_{dev,test}.tsv"""
import glob
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core   # noqa: E402

R = f"{ROOT}/results"
SYMS = ("intersect", "union", "gd", "gdf", "gdfa")
gold = {s: core.read_gold(f"{ROOT}/data/gold/{s}") for s in ("dev", "test")}


def align(al, sizes, corpus="parice"):
    from combalign import external
    os.makedirs(f"{R}/stat", exist_ok=True)
    src = gold["dev"][0] + gold["test"][0]
    trg = gold["dev"][1] + gold["test"][1]
    nd = len(gold["dev"][0])
    for size in sizes.split(","):
        n = None if size == "all" else int(size)
        tag = f"{corpus}{size}"
        work = f"{ROOT}/work/{al}.{tag}"
        fwd, rev = external.ALIGNERS[al](src, trg, f"{ROOT}/data/train/{corpus}.en", f"{ROOT}/data/train/{corpus}.is", n, work)
        with open(f"{R}/stat/{al}.{tag}.tsv", "w") as f:
            for sym in SYMS + ("fwd", "rev"):
                pred = fwd if sym == "fwd" else rev if sym == "rev" else [core.symmetrize(a, b, sym) for a, b in zip(fwd, rev)]
                for split, part in (("dev", pred[:nd]), ("test", pred[nd:])):
                    core.write_pharaoh(f"{R}/predictions/{split}.{al}.{tag}.{sym}.pharaoh", part)
                    e = core.evaluate(part, gold[split][2])
                    f.write(f"{split}\t{al}.{tag}.{sym}\t{e['P']}\t{e['R']}\t{e['F1']}\t{e['AER']}\n")
                    print(split, al, tag, sym, e, flush=True)
        os.system(f"rm -rf {work}")   # GIZA/eflomal work files are large


def combine():
    preds = {}
    for split in ("dev", "test"):
        for p in glob.glob(f"{R}/predictions/{split}.*.pharaoh"):
            preds[(split, os.path.basename(p)[len(split) + 1:-len(".pharaoh")])] = core.read_pharaoh(p)
    singles = {s: {n: core.evaluate(p, gold[s][2]) for (sp, n), p in preds.items() if sp == s} for s in ("dev", "test")}

    def best(prefix, sym="gdfa"):
        """best training size/corpus for an aligner, chosen on DEV."""
        c = [n for n in singles["dev"] if n.startswith(prefix + ".") and n.endswith("." + sym)]
        return max(c, key=lambda n: singles["dev"][n]["F1"]) if c else None

    def bestsym(prefix):
        c = [n for n in singles["dev"] if n.startswith(prefix + ".")]
        return max(c, key=lambda n: singles["dev"][n]["F1"]) if c else None

    G, FA, EF = best("giza"), best("fast_align"), best("eflomal")
    Gb, FAb, EFb = bestsym("giza"), bestsym("fast_align"), bestsym("eflomal")
    emb = [n for n in singles["dev"] if not n.split(".")[0] in ("giza", "fast_align", "eflomal")]
    E = sorted(emb, key=lambda n: -singles["dev"][n]["F1"])
    best_by_model = {}
    for n in E:
        best_by_model.setdefault(n.split(".")[0], n)
    combos = {
        "EnsSmall(2/3) paper": ([EF, FA, "xlmr.word.itermax.k0.0"], 2),
        "EnsLarge(3/5) paper": ([G, FA, EF, "mbert.word.argmax.k0.0", "xlmr.word.argmax.k0.0"], 3),
        "EnsLarge(3/5) best-sym": ([Gb, FAb, EFb, "mbert.word.argmax.k0.0", "xlmr.word.argmax.k0.0"], 3),
    }
    for k in (2, 3, 4):
        combos[f"stat3+bestemb({k}/{3 + len(best_by_model)})"] = ([Gb, FAb, EFb] + list(best_by_model.values()), k)
    for m in range(3, 8):
        top = sorted([x for x in (Gb, FAb, EFb) if x] + list(best_by_model.values()), key=lambda n: -singles["dev"][n]["F1"])[:m]
        for k in range(2, m):
            combos[f"top{m}-dev({k}/{m})"] = (top, k)
    for split in ("dev", "test"):
        rows = list(singles[split].items())
        for name, (members, k) in combos.items():
            if all(m and (split, m) in preds for m in members):
                rows.append((f"{name}: {' + '.join(members)}", core.evaluate(core.vote([preds[(split, m)] for m in members], k), gold[split][2])))
        with open(f"{R}/ensemble_{split}.tsv", "w") as f:
            f.write("system\tP\tR\tF1\tAER\tedges\n")
            for n, r in sorted(rows, key=lambda x: -x[1]["F1"]):
                f.write(f"{n}\t{r['P']}\t{r['R']}\t{r['F1']}\t{r['AER']}\t{r['edges']}\n")
    print(open(f"{R}/ensemble_test.tsv").read()[:3000])


if __name__ == "__main__":
    if sys.argv[1] == "align":
        align(*sys.argv[2:])
    else:
        combine()
