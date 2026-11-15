# -*- coding: utf-8 -*-
"""Batched multi-layer encoding (CPU, fp32): equals the per-sentence path, one pass serves several layers,
batch size and sentence order do not matter, and shared-pass members equal the frozen predictions."""
import os

import numpy as np
import pytest

from combalign import core
from conftest import fixture

pytestmark = pytest.mark.gold


@pytest.fixture(scope="module")
def labse():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from combalign import simalign
    try:
        return simalign.Embedder("labse", device="cpu")
    except Exception as e:
        pytest.skip(f"LaBSE unavailable: {e}")


def _same(a, b, tol=1e-5):
    return a[2] == b[2] and all(np.allclose(x, y, atol=tol) for x, y in ((a[0], b[0]), (a[1], b[1]), (a[3], b[3])))


def test_encode_equals_per_sentence_path_for_two_layers(gold, labse):
    sents = gold["test"][0][:12]
    got = labse.encode(sents, "en", layers=(6, 8), batch_size=5)
    for L in (6, 8):
        labse.layer = L
        assert all(_same(labse(s, "en"), g) for s, g in zip(sents, got[L])), L
    assert not np.allclose(got[6][0][0], got[8][0][0])               # the layers really differ


def test_batch_size_and_input_order_do_not_matter(gold, labse):
    sents = gold["test"][1][:10]
    a = labse.encode(sents, "is", (8,), batch_size=1)[8]
    b = labse.encode(sents, "is", (8,), batch_size=64)[8]
    rev = labse.encode(sents[::-1], "is", (8,), batch_size=3)[8][::-1]
    assert all(_same(x, y) and _same(x, z) for x, y, z in zip(a, b, rev))


def test_empty_batch_and_defaults(labse):
    assert labse.encode([], "en") == {labse.layer: []}


def test_shared_pass_members_equal_frozen_predictions_30_sentences(gold, labse):
    """LaBSE L8 and L6 κ.09 (two of the `full` members) from ONE encoder pass."""
    from combalign import simalign
    src, trg, _ = gold["test"]
    a8, a6 = simalign.align_corpus_multi(src[:30], trg[:30], [(8, "argmax", "word", 0.0), (6, "argmax", "word", 0.09)], labse,
                                         batch_size=8, chunk=11)
    assert a8 == fixture("test", "labse8")[:30]
    assert a6 == fixture("test", "labse6k09")[:30]


def test_align_corpus_with_cache_equals_without(gold, labse):
    from combalign import simalign
    src, trg, _ = gold["test"]
    cache = {}
    a = simalign.align_corpus(src[:10], trg[:10], "labse", embedder=labse, cache=cache, layer=8)
    b = simalign.align_corpus(src[:10], trg[:10], "labse", embedder=labse, layer=8)
    c = simalign.align_corpus(src[:10], trg[:10], "labse", method="itermax", embedder=labse, cache=cache, layer=8)
    assert a == b and len(cache) == 10 and len(c) == 10
