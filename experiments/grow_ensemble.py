# -*- coding: utf-8 -*-
"""Ensemble + grow: seed = vote ≥ k_seed over members, candidates = vote ≥ k_cand (or union of the
statistical gdfa outputs) → core.grow. Members = the dev-chosen top 7 (public data only) + phrase-capable aligners.
Writes predictions {split}.grow.{label}.pharaoh; then run mwe_eval.py."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core
R = f"{ROOT}/results"
TOP7 = ["labse8.word.argmax.k0.0", "labse.word.argmax.k0.09", "nllb.word.argmax.k0.0", "eflomal.pariceall.intersect",
        "nllb8.word.argmax.k0.09", "fast_align.pariceall.intersect", "giza.parice1000000.intersect"]
PHRASE = ["eflomal.pariceall.gdfa", "fast_align.pariceall.gdfa", "nllb.word.itermax.k0.09", "labse.word.itermax.k0.09",
          "nllb.bpe.softmax.k0.0"]
for split in ("dev", "test"):
    _, _, gold = core.read_gold(f"{ROOT}/data/gold/{split}")
    P = lambda n: core.read_pharaoh(f"{R}/predictions/{split}.{n}.pharaoh")   # noqa: E731
    top = [P(n) for n in TOP7]; ph = [P(n) for n in PHRASE]
    for ks in (3, 4):
        seed = core.vote(top, ks)
        for lab, cand in (("phrase2", core.vote(ph, 2)), ("phrase3", core.vote(ph, 3)), ("all2", core.vote(top + ph, 2)),
                          ("all3", core.vote(top + ph, 3))):
            for fa in (True, False):
                g = [core.grow(s, c, fa) for s, c in zip(seed, cand)]
                name = f"grow.top7v{ks}+{lab}{'.fa' if fa else ''}"
                core.write_pharaoh(f"{R}/predictions/{split}.{name}.pharaoh", g)
                print(split, name, core.evaluate(g, gold), flush=True)
