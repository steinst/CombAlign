# -*- coding: utf-8 -*-
"""End-to-end checks that recompute the embedding members (CPU, fp32) and compare them with the frozen predictions,
plus the statistical aligners on a small training set. Skipped without data/gold (models must be in the local
Hugging Face cache). Fast suite: first 30 test sentences; slow suite (`-m slow`): full dev + test, eflomal/fast_align."""
import os

import pytest

from combalign import core
from conftest import ROOT, fixture

pytestmark = pytest.mark.gold

# fixture name → (Embedder model, layer, κ)
EMBEDDERS = {"labse8": ("labse8", 0.0), "labse6k09": ("labse", 0.09), "nllb12": ("nllb", 0.0), "nllb8k09": ("nllb8", 0.09)}
_cache = {}


def _embedder(model):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from combalign import simalign
    if model not in _cache:
        try:
            _cache[model] = simalign.Embedder(model, device="cpu")
        except Exception as e:                                        # model not cached locally
            pytest.skip(f"model {model} unavailable: {e}")
    return _cache[model]


def _align(name, src, trg):
    from combalign import simalign
    model, kappa = EMBEDDERS[name]
    return simalign.align_corpus(src, trg, model, "argmax", "word", kappa, embedder=_embedder(model))


def _diff_links(a, b):
    return sum(len(x ^ y) for x, y in zip(a, b))


@pytest.mark.parametrize("name", ["labse8", "labse6k09", "nllb12"])
def test_embedding_member_equals_frozen_prediction_30_sentences(gold, name):
    src, trg, _ = gold["test"]
    out = _align(name, src[:30], trg[:30])
    assert _diff_links(out, fixture("test", name)[:30]) == 0


@pytest.mark.slow
@pytest.mark.parametrize("name", list(EMBEDDERS))
@pytest.mark.parametrize("split", ["dev", "test"])
def test_embedding_member_equals_frozen_prediction_full(gold, name, split):
    src, trg, _ = gold[split]
    out = _align(name, src, trg)
    # fp32 CPU reproduces the stored runs; allow nothing but report the number of differing links
    assert _diff_links(out, fixture(split, name)) == 0


@pytest.mark.slow
def test_emb_recipe_from_scratch_reproduces_reference(gold):
    """The `emb` recipe (3 encoders, vote 2/3) recomputed on the test set: F1 .947 / AER .054."""
    src, trg, g = gold["test"]
    preds = [_align(n, src, trg) for n in ("labse8", "labse6k09", "nllb12")]
    r = core.evaluate(core.vote(preds, 2), g)
    assert (r["F1"], r["AER"]) == (0.947, 0.054)


# stored ParIce-100k intersection runs (test set F1); the aligners are retrained, so allow a small band
@pytest.mark.slow
@pytest.mark.parametrize("aligner,ref", [("eflomal", 0.902), ("fast_align", 0.863)])
def test_statistical_aligner_on_100k_parice(gold, aligner, ref):
    from combalign import external
    train = f"{ROOT}/data/train/parice"
    if not os.path.exists(train + ".en"):
        pytest.skip("ParIce training data not available (data/train)")
    if aligner == "fast_align" and not os.path.exists(external.FA):
        pytest.skip("fast_align binary not built (install.sh)")
    src, trg, g = gold["test"]
    fwd, rev = external.ALIGNERS[aligner](src, trg, train + ".en", train + ".is", 100000, None)
    r = core.evaluate([core.symmetrize(f, b, "intersect") for f, b in zip(fwd, rev)], g)
    assert abs(r["F1"] - ref) < 0.01
