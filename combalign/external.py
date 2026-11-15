# -*- coding: utf-8 -*-
"""Wrappers for the statistical aligners (built by install.sh into ./tools; override with $COMBALIGN_TOOLS). Each trains on
training corpus + the sentences to align (unsupervised; the evaluation sentences are appended, as in the paper),
in both directions, and returns (fwd, rev) alignments for the last len(src) lines as sets of (src_i, trg_j).
Lowercased input. Symmetrise afterwards with core.symmetrize.
  fast_align   -d -o -v (Dyer et al. 2013); reverse with -r
  eflomal      eflomal.Aligner (Östling & Tiedemann 2016), forward + reverse links in one run
  giza         GIZA++ IBM1⁵ HMM⁵ IBM3³ IBM4³ (default), via plain2snt / snt2cooc / mkcls
"""
import os
import subprocess
import tempfile

from . import core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.environ.get("COMBALIGN_TOOLS", f"{ROOT}/tools")
FA = f"{TOOLS}/fast_align/build/fast_align"
GIZA = f"{TOOLS}/giza-pp/GIZA++-v2"
MKCLS = f"{TOOLS}/giza-pp/mkcls-v2/mkcls"


def _corpus(train_src, train_trg, src, trg, n_train):
    """Lines (lowercased, tokenised strings) = first n_train training pairs + the pairs to align."""
    S, T = [], []
    if train_src:
        with open(train_src, encoding="utf-8") as fs, open(train_trg, encoding="utf-8") as ft:
            for k, (a, b) in enumerate(zip(fs, ft)):
                if n_train is not None and k >= n_train:
                    break
                S.append(a.strip().lower()); T.append(b.strip().lower())
    S += [" ".join(s).lower() for s in src]
    T += [" ".join(t).lower() for t in trg]
    return S, T


def fast_align(src, trg, train_src=None, train_trg=None, n_train=None, work=None):
    S, T = _corpus(train_src, train_trg, src, trg, n_train)
    work = work or tempfile.mkdtemp(prefix="fa_")
    os.makedirs(work, exist_ok=True)
    p = f"{work}/corpus.fa"
    with open(p, "w", encoding="utf-8") as f:
        for a, b in zip(S, T):
            f.write(f"{a or '.'} ||| {b or '.'}\n")
    out = {}
    for d, flag in (("fwd", []), ("rev", ["-r"])):
        r = subprocess.run([FA, "-i", p, "-d", "-o", "-v"] + flag, capture_output=True, text=True, check=True)
        out[d] = [core.parse_pharaoh(l)[0] for l in r.stdout.splitlines()][-len(src):]
    return out["fwd"], out["rev"]


def eflomal(src, trg, train_src=None, train_trg=None, n_train=None, work=None):
    from eflomal import Aligner
    S, T = _corpus(train_src, train_trg, src, trg, n_train)
    work = work or tempfile.mkdtemp(prefix="ef_")
    os.makedirs(work, exist_ok=True)
    a = Aligner()
    a.align([s or "." for s in S], [t or "." for t in T], links_filename_fwd=f"{work}/fwd",
            links_filename_rev=f"{work}/rev", quiet=True)
    fwd = [core.parse_pharaoh(l)[0] for l in open(f"{work}/fwd")][-len(src):]
    rev = [core.parse_pharaoh(l)[0] for l in open(f"{work}/rev")][-len(src):]
    return fwd, rev


def _giza_dir(work, S, T, a, b, tag):
    """Run GIZA++ -S a -T b; returns per-line alignments as {(a_i, b_j)}."""
    fa, fb = f"{work}/{a}", f"{work}/{b}"
    out = f"{work}/{tag}"
    subprocess.run([f"{GIZA}/snt2cooc.out", f"{fa}.vcb", f"{fb}.vcb", f"{work}/{a}_{b}.snt"],
                   stdout=open(f"{work}/{a}_{b}.cooc", "w"), check=True)
    subprocess.run([f"{GIZA}/GIZA++", "-S", f"{fa}.vcb", "-T", f"{fb}.vcb", "-C", f"{work}/{a}_{b}.snt",
                    "-CoocurrenceFile", f"{work}/{a}_{b}.cooc", "-o", tag, "-outputpath", work,
                    "-sourcevocabularyclasses", f"{fa}.vcb.classes", "-targetvocabularyclasses", f"{fb}.vcb.classes"],
                   capture_output=True, check=True)
    # .A3.final: 3 lines per pair; line 3 = source words with "({ target positions })", 1-based, NULL first
    import glob
    import re
    res = []
    lines = open(glob.glob(f"{out}.A3.final*")[0], encoding="utf-8").read().splitlines()
    for k in range(0, len(lines), 3):
        al = set()
        for i, m in enumerate(re.finditer(r"\(\{([\d ]*)\}\)", lines[k + 2])):
            if i == 0:
                continue
            for j in m.group(1).split():
                al.add((i - 1, int(j) - 1))    # annotated word = a_i, listed positions = b_j
        res.append(al)
    return res


def giza(src, trg, train_src=None, train_trg=None, n_train=None, work=None):
    S, T = _corpus(train_src, train_trg, src, trg, n_train)
    work = work or tempfile.mkdtemp(prefix="giza_")
    os.makedirs(work, exist_ok=True)
    for name, lines in (("en", S), ("is", T)):
        with open(f"{work}/{name}", "w", encoding="utf-8") as f:
            f.write("\n".join(l or "." for l in lines) + "\n")
    subprocess.run([f"{GIZA}/plain2snt.out", f"{work}/en", f"{work}/is"], capture_output=True, check=True)
    for name in ("en", "is"):
        subprocess.run([MKCLS, "-n10", "-c50", f"-p{work}/{name}", f"-V{work}/{name}.vcb.classes"],
                       capture_output=True, check=True)
    # GIZA++ -S a -T b: .A3.final line 3 = a words annotated with b positions; each b word has ≤ 1 a word.
    en_is = _giza_dir(work, S, T, "en", "is", "en_is")   # (en_i, is_j)
    is_en = _giza_dir(work, S, T, "is", "en", "is_en")   # (is_i, en_j)
    fwd = [set(al) for al in en_is][-len(src):]
    rev = [{(j, i) for (i, j) in al} for al in is_en][-len(src):]
    return fwd, rev


ALIGNERS = {"fast_align": fast_align, "eflomal": eflomal, "giza": giza}
