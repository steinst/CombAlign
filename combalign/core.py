# -*- coding: utf-8 -*-
"""CombAlign core (Steingrímsson, Loftsson & Way, NoDaLiDa 2021). Everything works on token-index alignments {(i, j)} = source word i ↔ target word j.
  read_gold / write_pharaoh   Pharaoh format; the en-is gold set (CLARIN-IS) marks possible edges with "p" (3p4)
  evaluate                    precision vs possible (S ∪ P), recall vs sure, F1, AER (Och & Ney 2003)
  symmetrize                  intersection, union, grow-diag (gd), grow-diag-final (gdf), grow-diag-final-and (gdfa)
  vote                        ensemble: an edge is kept if ≥ k aligners propose it (intersectmin-k; k=1 is unionall);
                              return_counts=True also returns how many aligners proposed each kept edge
  grow                        add neighbouring candidate links to a seed; return_added=True also returns the added links
  to_scores / parse_scores    confidence file: one line per sentence, "i-j:c/n" = link proposed by c of n aligners,
                              "i-j:g3/5" = link added by grow, proposed by 3 of 5 candidate aligners,
                              "i-j:s" = seed link without a known count
"""
import re


def parse_pharaoh(line):
    sure, poss = set(), set()
    for tok in line.split():
        m = re.match(r"(\d+)([-p])(\d+)$", tok)
        if not m:
            continue
        e = (int(m.group(1)), int(m.group(3)))
        (sure if m.group(2) == "-" else poss).add(e)
    return sure, poss


def read_lines(path):
    """Sentence files are "<id>\\t<tokenised sentence>"; returns the token lists."""
    out = []
    for l in open(path, encoding="utf-8"):
        l = l.rstrip("\n")
        out.append((l.split("\t", 1)[1] if "\t" in l else l).split())
    return out


def read_raw(path):
    """→ (ids, texts): per line the id (None if the line has no "<id>\\t" prefix) and the sentence as it stands."""
    ids, texts = [], []
    for l in open(path, encoding="utf-8"):
        sid, tab, text = l.rstrip("\n").partition("\t") if "\t" in l else (None, "", l.rstrip("\n"))
        ids.append(sid)
        texts.append(text)
    return ids, texts


def read_gold(prefix):
    """prefix e.g. data/gold/test → (src tokens, trg tokens, [(sure, possible)])."""
    src = read_lines(prefix + "_sentences.en")
    trg = read_lines(prefix + "_sentences.is")
    gold = [parse_pharaoh(l) for l in open(prefix + "_alignments.pharaoh", encoding="utf-8")]
    return src, trg, gold


def to_pharaoh(al):
    return " ".join(f"{i}-{j}" for i, j in sorted(al))


def write_pharaoh(path, alignments):
    with open(path, "w", encoding="utf-8") as f:
        for al in alignments:
            f.write(to_pharaoh(al) + "\n")


def read_pharaoh(path):
    return [parse_pharaoh(l)[0] | parse_pharaoh(l)[1] for l in open(path, encoding="utf-8")]


def evaluate(pred, gold):
    """pred: [set], gold: [(sure, possible)]; possible includes sure for precision (standard practice)."""
    a = s = p_hit = s_hit = 0
    for A, (S, P) in zip(pred, gold):
        PS = S | P
        a += len(A)
        s += len(S)
        p_hit += len(A & PS)
        s_hit += len(A & S)
    prec, rec = p_hit / max(1, a), s_hit / max(1, s)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    aer = 1 - (s_hit + p_hit) / max(1, a + s)
    return {"P": round(prec, 3), "R": round(rec, 3), "F1": round(f1, 3), "AER": round(aer, 3), "edges": a}


def symmetrize(fwd, rev, how):
    """fwd: {(i, j)} from src→trg, rev: {(i, j)} from trg→src (already in (src, trg) order)."""
    if how == "intersect":
        return fwd & rev
    if how == "union":
        return fwd | rev
    al = set(fwd & rev)
    union = fwd | rev
    nb = [(-1, 0), (0, -1), (1, 0), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if how in ("gd", "gdf", "gdfa") else [])
    changed = True
    while changed:                                    # grow-diag
        changed = False
        for (i, j) in sorted(al):
            for di, dj in nb:
                n = (i + di, j + dj)
                if n in union and n not in al and (all(x[0] != n[0] for x in al) or all(x[1] != n[1] for x in al)):
                    al.add(n)
                    changed = True
    if how in ("gdf", "gdfa"):                         # final(-and)
        for d in (fwd, rev):
            for n in sorted(d):
                si = all(x[0] != n[0] for x in al)
                ti = all(x[1] != n[1] for x in al)
                if (how == "gdf" and (si or ti)) or (how == "gdfa" and si and ti):
                    al.add(n)
    return al


def vote(alignment_lists, k, return_counts=False):
    """alignment_lists: [aligner][sentence] → set; keep edges proposed by ≥ k aligners.
    return_counts=True → per sentence {edge: number of aligners that proposed it} for the kept edges (a dict, so
    `set(d)` is the plain result); default → per sentence a set."""
    out = []
    for sets in zip(*alignment_lists):
        cnt = {}
        for s in sets:
            for e in s:
                cnt[e] = cnt.get(e, 0) + 1
        out.append({e: c for e, c in cnt.items() if c >= k} if return_counts else {e for e, c in cnt.items() if c >= k})
    return out


def grow(seed, cand, final_and=True, return_added=False):
    """grow-diag(-final-and) generalised to an ensemble: start from `seed` (e.g. a high-precision vote), add
    neighbouring links from `cand` (e.g. links proposed by ≥ 2 aligners) whose source or target word is still
    unaligned, then (final-and) any cand link with both words unaligned. Lets 1-1 ensembles form 1-n / n-m blocks.
    return_added=True → (alignment, set of the links grow added), the rest of the alignment being the seed."""
    al = set(seed)
    nb = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    changed = True
    while changed:
        changed = False
        for (i, j) in sorted(al):
            for di, dj in nb:
                n = (i + di, j + dj)
                if n in cand and n not in al and (all(x[0] != n[0] for x in al) or all(x[1] != n[1] for x in al)):
                    al.add(n)
                    changed = True
    if final_and:
        for n in sorted(cand):
            if all(x[0] != n[0] for x in al) and all(x[1] != n[1] for x in al):
                al.add(n)
    return (al, al - set(seed)) if return_added else al


def to_scores(votes, n, grown=None, grown_counts=None, m=None):
    """One confidence line. votes: {edge: c} for the voted links (c of n aligners; c=None → "i-j:s", count unknown);
    grown: set of links added by grow with grown_counts {edge: c} out of m candidate aligners."""
    toks = []
    for (i, j), c in votes.items():
        toks.append((i, j, f"{i}-{j}:s" if c is None else f"{i}-{j}:{c}/{n}"))
    for (i, j) in (grown or ()):
        toks.append((i, j, f"{i}-{j}:g{grown_counts[(i, j)]}/{m}" if grown_counts and (i, j) in grown_counts
                     else f"{i}-{j}:g"))
    return " ".join(t for _, _, t in sorted(toks))


def parse_scores(line):
    """→ {(i, j): (c, n, grown)}; c, n are None for "i-j:s" and for a grown link without a count."""
    out = {}
    for tok in line.split():
        m = re.match(r"(\d+)-(\d+):(?:(s)|(g)?(?:(\d+)/(\d+))?)$", tok)
        if not m:
            raise ValueError(f"bad score token: {tok!r}")
        c = int(m.group(5)) if m.group(5) else None
        n = int(m.group(6)) if m.group(6) else None
        out[(int(m.group(1)), int(m.group(2)))] = (c, n, m.group(4) == "g")
    return out


def filter_scores(parsed, min_votes=1, keep_grown=True):
    """parsed: parse_scores output → link set: voted links with count ≥ min_votes (links without a count are kept),
    grown links only if keep_grown."""
    return {e for e, (c, n, g) in parsed.items() if (keep_grown if g else (c is None or c >= min_votes))}
