# -*- coding: utf-8 -*-
"""One row per aligner: configuration chosen on DEV (best F1), reported on DEV and TEST (independent of test).
→ results/SINGLE_ALIGNERS.md"""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import core
R = f"{ROOT}/results"
G = {s: core.read_gold(f"{ROOT}/data/gold/{s}")[2] for s in ("dev", "test")}
FAM = [("GIZA++ (ParIce)", "giza."), ("fast_align", "fast_align."), ("eflomal", "eflomal."),
       ("SimAlign mBERT", "mbert."), ("SimAlign XLM-R base", "xlmr."), ("XLM-R large", "xlmrL."),
       ("multilingual-e5", "e5."), ("LaBSE", "labse"), ("NLLB-600M encoder (TransAlign)", "nllb"),
       ("grow ensemble (ours)", "grow.")]
L = ["# Single aligners on the en-is gold set (configuration chosen on dev, then scored on test)", "",
     "| aligner | chosen configuration | dev F1 | test P | test R | test F1 | test AER |", "|---|---|---|---|---|---|---|"]
for lab, pre in FAM:
    names = [os.path.basename(p)[4:-8] for p in glob.glob(f"{R}/predictions/dev.{pre}*.pharaoh")]
    names = [n for n in names if not n.startswith("grow.") or pre == "grow."]
    names = [n for n in names if os.path.exists(f"{R}/predictions/test.{n}.pharaoh")]
    if not names:
        continue
    dev = {n: core.evaluate(core.read_pharaoh(f"{R}/predictions/dev.{n}.pharaoh"), G["dev"]) for n in names}
    b = max(names, key=lambda n: dev[n]["F1"])
    t = core.evaluate(core.read_pharaoh(f"{R}/predictions/test.{b}.pharaoh"), G["test"])
    L.append(f"| {lab} | {b} | {dev[b]['F1']:.3f} | {t['P']:.3f} | {t['R']:.3f} | {t['F1']:.3f} | {t['AER']:.3f} |")
open(f"{R}/SINGLE_ALIGNERS.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
