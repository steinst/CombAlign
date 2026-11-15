# -*- coding: utf-8 -*-
"""Shared helpers. The gold set is not in the repo: tests that need it are skipped when data/gold is missing.
Run:  pytest tests/          fast suite (seconds to ~2 min on CPU)
      pytest tests/ -m slow  slow suite (full dev/test with the encoders on CPU, eflomal)
      pytest tests/ -m ""    everything"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # tests are CPU-only and deterministic
FIXTURES = f"{ROOT}/tests/fixtures"
GOLD = f"{ROOT}/data/gold"

from combalign import core   # noqa: E402

# the 7 `full` members (fixture names), and the recipes built from them; vote k per recipe
FULL = ["labse8", "labse6k09", "nllb12", "nllb8k09", "eflomal", "fast_align", "giza"]
GROW_MEMBERS = ["eflomal_gdfa", "fast_align_gdfa", "nllb_itermax", "labse_itermax", "nllb_softmax"]
RECIPES = {"emb": (["labse8", "labse6k09", "nllb12"], 2),
           "fast": (["labse6k09", "nllb12", "eflomal"], 2),
           "full": (FULL, 3)}
# reference numbers: split → F1 / AER, to the 3 decimals the paper tables use
REF = {"emb": {"dev": (0.958, 0.043), "test": (0.947, 0.054)},
       "fast": {"dev": (0.966, 0.035), "test": (0.954, 0.046)},
       "full": {"dev": (0.967, 0.036), "test": (0.960, 0.043)}}


def fixture(split, name):
    return core.read_pharaoh(f"{FIXTURES}/{split}.{name}.pharaoh")


@pytest.fixture(scope="session")
def gold():
    if not os.path.exists(f"{GOLD}/test_alignments.pharaoh"):
        pytest.skip("gold set not available (data/gold)")
    return {s: core.read_gold(f"{GOLD}/{s}") for s in ("dev", "test")}
