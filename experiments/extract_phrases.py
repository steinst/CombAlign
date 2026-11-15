# -*- coding: utf-8 -*-
"""Phrase-pair and word-pair counts from a whole training corpus (MWEs, bilingual dictionary induction).
  align    eflomal on all of data/train/<corpus>.{en,is} (lowercased) → work/phr/<corpus>.{fwd,rev}
  count    gdfa per line (eflomal gdfa: 39 % of the gold set's MWE blocks exact, see results/RESULTS.md) → connected blocks;
           keep blocks contiguous on both sides, ≤ 4 words per side, at least one alphabetic word per side;
           counts of (en, is) block pairs + marginal counts of every en/is n-gram (n ≤ 4) for Dice
           → work/phr/<corpus>.pairs.tsv (en, is, count, en_count, is_count, dice), count ≥ 3
    python experiments/extract_phrases.py align|count <corpus> [procs=8]"""
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core  # noqa: E402

W = f"{ROOT}/work/phr"
ALPHA = re.compile(r"[^\W\d_]", re.U)


def align(corpus):
    from eflomal import Aligner
    os.makedirs(W, exist_ok=True)
    S = [l.strip().lower() or "." for l in open(f"{ROOT}/data/train/{corpus}.en", encoding="utf-8")]
    T = [l.strip().lower() or "." for l in open(f"{ROOT}/data/train/{corpus}.is", encoding="utf-8")]
    Aligner().align(S, T, links_filename_fwd=f"{W}/{corpus}.fwd", links_filename_rev=f"{W}/{corpus}.rev", quiet=True)
    print("aligned", len(S))


def blocks(al):
    adj = {}
    for i, j in al:
        adj.setdefault(("s", i), set()).add(("t", j))
        adj.setdefault(("t", j), set()).add(("s", i))
    seen = set()
    for n in adj:
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x); comp.add(x); stack.extend(adj[x])
        yield sorted(i for k, i in comp if k == "s"), sorted(j for k, j in comp if k == "t")


def ngrams(toks, n=4):
    for a in range(len(toks)):
        for b in range(a + 1, min(len(toks), a + n) + 1):
            g = toks[a:b]
            if any(ALPHA.search(w) for w in g):
                yield " ".join(g)


def work(args):
    """pass 1 (keep=None): block pairs. pass 2 (keep=(en set, is set)): marginal counts of those phrases only."""
    corpus, start, stop, keep = args
    pairs, ce, ci = Counter(), Counter(), Counter()
    with open(f"{ROOT}/data/train/{corpus}.en", encoding="utf-8") as fe, \
         open(f"{ROOT}/data/train/{corpus}.is", encoding="utf-8") as fi, \
         open(f"{W}/{corpus}.fwd") as ff, open(f"{W}/{corpus}.rev") as fr:
        for k, (e, i, f, r) in enumerate(zip(fe, fi, ff, fr)):
            if k < start:
                continue
            if k >= stop:
                break
            E, I = e.lower().split(), i.lower().split()
            if keep is not None:
                ce.update(g for g in set(ngrams(E)) if g in keep[0])
                ci.update(g for g in set(ngrams(I)) if g in keep[1])
                continue
            al = core.symmetrize(core.parse_pharaoh(f)[0], core.parse_pharaoh(r)[0], "gdfa")
            for bs, bt in blocks(al):
                if len(bs) > 4 or len(bt) > 4 or bs[-1] - bs[0] + 1 != len(bs) or bt[-1] - bt[0] + 1 != len(bt):
                    continue
                pe = " ".join(E[x] for x in bs if x < len(E)); pi = " ".join(I[x] for x in bt if x < len(I))
                if ALPHA.search(pe) and ALPHA.search(pi):
                    pairs[(pe, pi)] += 1
    return pairs, ce, ci


_KEEP = None


def _init(keep):
    global _KEEP
    _KEEP = keep


def _work2(args):
    return work(args + (_KEEP,))


def count(corpus, procs=8):
    n = sum(1 for _ in open(f"{ROOT}/data/train/{corpus}.en", encoding="utf-8"))
    step = n // procs + 1
    chunks = [(corpus, s, s + step) for s in range(0, n, step)]
    pairs = Counter()
    with Pool(procs) as p:
        for a, _, _ in p.imap_unordered(work, [c + (None,) for c in chunks]):
            pairs.update(a)
    pairs = Counter({k: v for k, v in pairs.items() if v >= 3})
    keep = ({e for e, _ in pairs}, {i for _, i in pairs})
    print("pass 1:", len(pairs), "pair types with count >= 3", flush=True)
    ce, ci = Counter(), Counter()
    with Pool(procs, initializer=_init, initargs=(keep,)) as p:
        for _, b, c in p.imap_unordered(_work2, chunks):
            ce.update(b); ci.update(c)
    with open(f"{W}/{corpus}.pairs.tsv", "w", encoding="utf-8") as f:
        f.write("en\tis\tcount\ten_count\tis_count\tdice\n")
        for (e, i), c in pairs.most_common():
            f.write(f"{e}\t{i}\t{c}\t{ce[e]}\t{ci[i]}\t{2 * c / max(1, ce[e] + ci[i]):.4f}\n")
    print(corpus, n, "lines", len(pairs), "pair types")


if __name__ == "__main__":
    if sys.argv[1] == "align":
        align(sys.argv[2])
    else:
        count(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
