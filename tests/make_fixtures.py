# -*- coding: utf-8 -*-
"""Freeze the regression fixtures: copies of the stored dev/test predictions of the `full` members and of the
grow members (the 7 members + 5 phrase-capable ones), plus `grow_full` / `grow_full_fa`: snapshots of what the
2.0 core produces (seed = vote 4/7, candidates = vote 3/5 of the grow members, final-and off / on). The original
experiment's grow files are NOT used: they cannot be reproduced from these public members (see TOOL_STATUS.md).
Run once (the fixtures are committed):
    python tests/make_fixtures.py PREDICTIONS_DIR
Fixtures are predictions only (ParIce-trained statistical aligners, untrained encoders); the gold set is not in the repo."""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combalign import core   # noqa: E402

MEMBERS = {   # fixture name → stored prediction name
    "labse8": "labse8.word.argmax.k0.0",
    "labse6k09": "labse.word.argmax.k0.09",
    "nllb12": "nllb.word.argmax.k0.0",
    "nllb8k09": "nllb8.word.argmax.k0.09",
    "eflomal": "eflomal.pariceall.intersect",
    "fast_align": "fast_align.pariceall.intersect",
    "giza": "giza.parice1000000.intersect",
    "eflomal_gdfa": "eflomal.pariceall.gdfa",
    "fast_align_gdfa": "fast_align.pariceall.gdfa",
    "nllb_itermax": "nllb.word.itermax.k0.09",
    "labse_itermax": "labse.word.itermax.k0.09",
    "nllb_softmax": "nllb.bpe.softmax.k0.0",
}

FULL = ["labse8", "labse6k09", "nllb12", "nllb8k09", "eflomal", "fast_align", "giza"]
GROW_MEMBERS = ["eflomal_gdfa", "fast_align_gdfa", "nllb_itermax", "labse_itermax", "nllb_softmax"]

if __name__ == "__main__":
    src = sys.argv[1]
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    os.makedirs(dst, exist_ok=True)
    for split in ("dev", "test"):
        for name, stored in MEMBERS.items():
            shutil.copy(f"{src}/{split}.{stored}.pharaoh", f"{dst}/{split}.{name}.pharaoh")
        P = lambda n: core.read_pharaoh(f"{dst}/{split}.{n}.pharaoh")   # noqa: E731
        seed = core.vote([P(n) for n in FULL], 4)
        cand = core.vote([P(n) for n in GROW_MEMBERS], 3)
        for tag, fa in (("grow_full", False), ("grow_full_fa", True)):
            core.write_pharaoh(f"{dst}/{split}.{tag}.pharaoh", [core.grow(a, b, fa) for a, b in zip(seed, cand)])
