# -*- coding: utf-8 -*-
"""Calibration of the vote count as a confidence score (dev and test, `full` recipe = 7 members).
For every count c of the 7 members: how many links were proposed by exactly c members and which share of them is
correct (in sure ∪ possible); and the trade-off when keeping only links with ≥ c votes (--min-votes): P, R, F1, AER.
Also the precision of links added by grow, by the number of the 5 candidate aligners that proposed them.
    python experiments/calibration.py [PREDICTIONS_DIR=tests/fixtures] [--write]
--write replaces the section between the calibration markers in results/RESULTS.md (adds it at the end if missing)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core   # noqa: E402

FULL = ["labse8", "labse6k09", "nllb12", "nllb8k09", "eflomal", "fast_align", "giza"]
GROW_MEMBERS = ["eflomal_gdfa", "fast_align_gdfa", "nllb_itermax", "labse_itermax", "nllb_softmax"]
BEGIN, END = "<!-- calibration:begin (experiments/calibration.py) -->", "<!-- calibration:end -->"


def tables(pred_dir):
    out = {}
    for split in ("dev", "test"):
        gold = core.read_gold(f"{ROOT}/data/gold/{split}")[2]
        ok = [S | P for S, P in gold]
        P = lambda n: core.read_pharaoh(f"{pred_dir}/{split}.{n}.pharaoh")   # noqa: E731
        counts = core.vote([P(m) for m in FULL], 1, return_counts=True)
        by = {c: [0, 0] for c in range(1, len(FULL) + 1)}
        for d, good in zip(counts, ok):
            for e, c in d.items():
                by[c][1] += 1
                by[c][0] += e in good
        cum = {c: core.evaluate([{e for e, v in d.items() if v >= c} for d in counts], gold) for c in by}
        seed = core.vote([P(m) for m in FULL], 4)
        cand = core.vote([P(m) for m in GROW_MEMBERS], 3, return_counts=True)
        gro = {c: [0, 0] for c in range(3, len(GROW_MEMBERS) + 1)}
        for s, cd, good in zip(seed, cand, ok):
            _, added = core.grow(s, set(cd), False, return_added=True)
            for e in added:
                gro[cd[e]][1] += 1
                gro[cd[e]][0] += e in good
        out[split] = (by, cum, gro)
    return out


def render(t):
    L = ["### Confidence: precision by vote count", "",
         "Share of links (vs sure ∪ possible) that are correct, by how many of the 7 `full` members proposed the link, and the",
         "result of keeping only links with at least that many votes (`--min-votes`). Computed by `experiments/calibration.py`",
         "from the frozen predictions.", "",
         "| votes of 7 | dev links | dev precision | test links | test precision | test ≥ votes: P | R | F1 | AER |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in range(1, 8):
        (dh, dn), (th, tn), r = t["dev"][0][c], t["test"][0][c], t["test"][1][c]
        L.append(f"| {c} | {dn} | {dh / max(1, dn):.3f} | {tn} | {th / max(1, tn):.3f} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} | {r['AER']:.3f} |")
    L += ["", "Links added by grow (seed = vote ≥ 4/7, candidates = ≥ 3 of 5 phrase-capable aligners, final-and off), by the",
          "number of the 5 candidate aligners that proposed them:", "",
          "| candidate votes of 5 | dev links | dev precision | test links | test precision |", "|---|---|---|---|---|"]
    for c in range(3, 6):
        (dh, dn), (th, tn) = t["dev"][2][c], t["test"][2][c]
        L.append(f"| {c} | {dn} | {dh / max(1, dn):.3f} | {tn} | {th / max(1, tn):.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    text = render(tables(args[0] if args else f"{ROOT}/tests/fixtures"))
    print(text)
    if "--write" in sys.argv:
        path = f"{ROOT}/results/RESULTS.md"
        doc = open(path, encoding="utf-8").read()
        block = f"{BEGIN}\n{text}{END}\n"
        if BEGIN in doc:
            doc = doc[:doc.index(BEGIN)] + block + doc[doc.index(END) + len(END) + 1:]
        else:
            doc = doc.rstrip("\n") + "\n\n" + block
        open(path, "w", encoding="utf-8").write(doc)
