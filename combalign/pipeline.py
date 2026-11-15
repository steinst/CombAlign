# -*- coding: utf-8 -*-
"""`combalign align`: run a recipe end to end.
 * encoder members of one model share forward passes (one pass serves LaBSE L6 + L8, NLLB L8 + L12);
 * statistical aligners run in worker threads (external processes) while the encoders run;
 * every member's output is cached in the work directory, so an interrupted run resumes where it stopped and a rerun
   with other options (--scores, --min-votes, --grow) costs nothing. The cache key covers the member, both input
   files, the training files, the languages and the precision (fp16/fp32); a changed input recomputes the member.
"""
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from . import core
from .recipes import MEMBERS, RECIPES


def _file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Cache:
    """work/<member>.pharaoh + work/<member>.key."""

    def __init__(self, work, base_key):
        self.work, self.base = work, base_key
        os.makedirs(work, exist_ok=True)

    def key(self, name):
        spec = MEMBERS[name]
        # statistical members: the training files, identified by path, size and mtime (they can be huge);
        # encoder members: the precision (a statistical member is not recomputed when only --fp32 or the device changes)
        own = "train" if spec["kind"] == "stat" else "precision"
        base = {k: v for k, v in self.base.items() if k not in ("train", "precision")}
        return hashlib.sha1(json.dumps([base, name, spec, self.base.get(own)],
                                       sort_keys=True, default=str).encode()).hexdigest()

    def get(self, name):
        p, k = f"{self.work}/{name}.pharaoh", f"{self.work}/{name}.key"
        if os.path.exists(p) and os.path.exists(k) and open(k).read() == self.key(name):
            return core.read_pharaoh(p)
        return None

    def put(self, name, als):
        core.write_pharaoh(f"{self.work}/{name}.pharaoh", als)
        with open(f"{self.work}/{name}.key", "w") as f:
            f.write(self.key(name))


def _run_stat(aligner, src, trg, train_src, train_trg, n_train, work, syms):
    from . import external
    fwd, rev = external.ALIGNERS[aligner](src, trg, train_src, train_trg, n_train, work)
    return {sym: [core.symmetrize(f, r, sym) for f, r in zip(fwd, rev)] for sym in syms}


def run(recipe, src, trg, train_src=None, train_trg=None, src_lang="en", trg_lang="is", grow=False, device=None,
        fp16=None, work=None, batch_size=64, min_votes=None, log=None):
    """src, trg: lists of token lists. min_votes: raise the vote threshold (of the vote, or of grow's seed) to at least this.
    → dict(alignments=[set], counts=[{edge: votes}] (every proposed link), n=members voting, grown=[set of links added by grow]
    or None, grown_counts=[{edge: c}] or None, m=grow members)."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    R = RECIPES[recipe]
    if R["needs_corpus"] and not (train_src and train_trg):
        raise SystemExit(f"recipe '{recipe}' trains statistical aligners and needs --train-src/--train-trg (a parallel corpus). "
                         f"Without one use --recipe emb (encoders only, no training).")
    if len(src) != len(trg):
        raise SystemExit(f"--src has {len(src)} lines but --trg has {len(trg)}")
    names = list(R["members"]) + (list(R["grow"]["members"]) if grow else [])
    work = work or tempfile.mkdtemp(prefix=f"combalign_{recipe}_")
    base = {"src": hash_of(src), "trg": hash_of(trg), "src_lang": src_lang, "trg_lang": trg_lang}
    if any(MEMBERS[n]["kind"] == "embed" for n in names):
        # encoder precision: fp16 on GPU by default; CPU is always fp32 (links differ by a handful at most)
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        base["precision"] = "fp16" if device.startswith("cuda") and fp16 is not False else "fp32"
    stat_names = [n for n in names if MEMBERS[n]["kind"] == "stat"]
    if stat_names:
        base["train"] = [[p, os.path.getsize(p), os.stat(p).st_mtime_ns] for p in (train_src, train_trg)]
    cache = Cache(work, base)
    got = {n: cache.get(n) for n in names}
    for n, v in got.items():
        if v is not None:
            log(f"[cache] {n}")
    todo = [n for n in names if got[n] is None]

    # statistical aligners: one training run per aligner serves all its symmetrisations; run in worker threads
    pool, futures = ThreadPoolExecutor(max_workers=max(1, len(stat_names))), {}
    for aligner in dict.fromkeys(MEMBERS[n]["aligner"] for n in todo if MEMBERS[n]["kind"] == "stat"):
        mine = [n for n in todo if MEMBERS[n]["kind"] == "stat" and MEMBERS[n]["aligner"] == aligner]
        n_train = MEMBERS[mine[0]]["n_train"]
        log(f"[start] {aligner} (CPU, in the background): {', '.join(mine)}")
        futures[aligner] = (mine, pool.submit(_run_stat, aligner, src, trg, train_src, train_trg, n_train,
                                              f"{work}/{aligner}.tmp", sorted({MEMBERS[n]["sym"] for n in mine})))

    # encoders: one embedder per model, all its members from shared passes
    from . import simalign
    models = list(dict.fromkeys(MEMBERS[n]["model"] for n in todo if MEMBERS[n]["kind"] == "embed"))
    for model in models:
        mine = [n for n in todo if MEMBERS[n]["kind"] == "embed" and MEMBERS[n]["model"] == model]
        specs = [(MEMBERS[n]["layer"], MEMBERS[n]["method"], MEMBERS[n]["level"], MEMBERS[n]["kappa"]) for n in mine]
        log(f"[start] {model} layers {sorted({s[0] for s in specs})}: {', '.join(mine)}")
        E = simalign.Embedder(model, device=device, fp16=fp16)
        for n, als in zip(mine, simalign.align_corpus_multi(src, trg, specs, E, src_lang, trg_lang, batch_size)):
            got[n] = als
            cache.put(n, als)
            log(f"[done] {n}")
        del E

    for aligner, (mine, fut) in futures.items():
        res = fut.result()
        for n in mine:
            got[n] = res[MEMBERS[n]["sym"]]
            cache.put(n, got[n])
            log(f"[done] {n}")
    pool.shutdown()

    members = [got[n] for n in R["members"]]
    out = {"n": len(members), "grown": None, "grown_counts": None, "m": None}
    out["counts"] = core.vote(members, 1, return_counts=True)       # every proposed link with its vote count
    if not grow:
        out["alignments"] = core.vote(members, max(R["k"], min_votes or 0))
        return out
    G = R["grow"]
    cand = core.vote([got[n] for n in G["members"]], G["cand_k"], return_counts=True)
    seed = core.vote(members, max(G["seed_k"], min_votes or 0))
    res = [core.grow(s, set(c), False, return_added=True) for s, c in zip(seed, cand)]
    out.update(alignments=[a for a, _ in res], grown=[g for _, g in res], grown_counts=cand, m=len(G["members"]), seed=seed)
    return out


def hash_of(sentences):
    return hashlib.sha1("\n".join(" ".join(s) for s in sentences).encode()).hexdigest()
