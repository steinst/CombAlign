# -*- coding: utf-8 -*-
"""Multiword expressions in the en-is gold set. Gold blocks = connected components of the gold links
(sure ∪ possible) per sentence; a block is an MWE block if it has ≥ 2 words on at least one side (1-n, n-1, n-m).
For each prediction file (results/predictions/{split}.*.pharaoh):
  link recall on 1-1 blocks vs on MWE blocks (sure ∪ possible links of the block)
  block exact: predicted links restricted to the block's words reproduce the block's word sets on both sides
              (every word of the block linked, no link leaving the block) — the property phrase extraction needs
  block covered: all words of the block linked to something inside the block (looser)
    python experiments/mwe_eval.py [split=test] → results/mwe_{split}.tsv + results/mwe_examples_{split}.txt"""
import glob
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core  # noqa: E402


def components(links):
    adj = {}
    for i, j in links:
        adj.setdefault(("s", i), set()).add(("t", j))
        adj.setdefault(("t", j), set()).add(("s", i))
    seen, out = set(), []
    for n in adj:
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x); stack.extend(adj[x])
        S = frozenset(i for k, i in comp if k == "s")
        T = frozenset(j for k, j in comp if k == "t")
        out.append((S, T))
    return out


def main(split="test"):
    src, trg, gold = core.read_gold(f"{ROOT}/data/gold/{split}")
    blocks = []                                   # (sent, S, T, sure links in block, is_mwe)
    for k, (S, P) in enumerate(gold):
        for bs, bt in components(S | P):
            sure = {(i, j) for i, j in S | P if i in bs and j in bt}   # MWE links are mostly marked "possible"
            blocks.append((k, bs, bt, sure, len(bs) > 1 or len(bt) > 1))
    mwe = [b for b in blocks if b[4]]
    ex = Counter((" ".join(src[k][i] for i in sorted(bs)).lower(), " ".join(trg[k][j] for j in sorted(bt)).lower())
                 for k, bs, bt, _, m in mwe if m)
    with open(f"{ROOT}/results/mwe_examples_{split}.txt", "w") as f:
        f.write(f"{len(mwe)} MWE blocks of {len(blocks)} gold blocks; shapes "
                f"{Counter(('1' if len(b[1]) == 1 else 'n') + '-' + ('1' if len(b[2]) == 1 else 'n') for b in mwe)}\n")
        for (e, i), c in ex.most_common():
            f.write(f"{c}\t{e}\t{i}\n")
    systems = {os.path.basename(p)[len(split) + 1:-8]: core.read_pharaoh(p)
               for p in sorted(glob.glob(f"{ROOT}/results/predictions/{split}.*.pharaoh"))}
    # ensembles from the combine step: "label(k/m): a + b + c"
    import re
    if os.path.exists(f"{ROOT}/results/ensemble_{split}.tsv"):
        for l in list(open(f"{ROOT}/results/ensemble_{split}.tsv"))[1:]:
            name = l.split("\t")[0]
            m = re.match(r"(.*)\((\d+)/\d+\).*?: (.*)$", name)
            if m and all(x in systems for x in m.group(3).split(" + ")):
                systems[f"VOTE {m.group(1).strip()}({m.group(2)})"] = core.vote([systems[x] for x in m.group(3).split(" + ")], int(m.group(2)))
    rows = []
    for name, pred in systems.items():
        r11 = [0, 0]; rm = [0, 0]; exact = covered = 0
        for k, bs, bt, sure, is_m in blocks:
            A = pred[k]
            hit = len(sure & A)
            (rm if is_m else r11)[0] += hit
            (rm if is_m else r11)[1] += len(sure)
            if not is_m:
                continue
            inside = {(i, j) for i, j in A if i in bs and j in bt}
            leaving = {(i, j) for i, j in A if (i in bs) != (j in bt)}
            ls = {i for i, _ in inside}; lt = {j for _, j in inside}
            if ls == set(bs) and lt == set(bt):
                covered += 1
                if not leaving:
                    exact += 1
        rows.append((name, r11[0] / max(1, r11[1]), rm[0] / max(1, rm[1]), covered / len(mwe), exact / len(mwe)))
    with open(f"{ROOT}/results/mwe_{split}.tsv", "w") as f:
        f.write("system\trecall_1to1\trecall_mwe\tmwe_block_covered\tmwe_block_exact\n")
        for r in sorted(rows, key=lambda x: -x[4]):
            f.write(f"{r[0]}\t{r[1]:.3f}\t{r[2]:.3f}\t{r[3]:.3f}\t{r[4]:.3f}\n")
    print(open(f"{ROOT}/results/mwe_examples_{split}.txt").readline())


if __name__ == "__main__":
    main(*sys.argv[1:])
