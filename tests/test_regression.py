# -*- coding: utf-8 -*-
"""Gold-set regression checks on the frozen predictions (tests/fixtures). Fast: vote/grow/evaluate only, no models.
Skipped without data/gold."""
import pytest

from combalign import core
from conftest import FULL, GROW_MEMBERS, RECIPES, REF, fixture

pytestmark = pytest.mark.gold


def _vote(recipe, split):
    members, k = RECIPES[recipe]
    return core.vote([fixture(split, m) for m in members], k)


@pytest.mark.parametrize("split", ["dev", "test"])
@pytest.mark.parametrize("recipe", ["emb", "fast", "full"])
def test_recipe_votes_reproduce_reference_numbers(gold, recipe, split):
    r = core.evaluate(_vote(recipe, split), gold[split][2])
    assert (r["F1"], r["AER"]) == REF[recipe][split]


def test_full_recipe_edge_count_is_frozen(gold):
    assert core.evaluate(_vote("full", "test"), gold["test"][2])["edges"] == 4691
    assert core.evaluate(_vote("full", "dev"), gold["dev"][2])["edges"] == 2280


@pytest.mark.parametrize("split", ["dev", "test"])
def test_fixture_sentence_counts_match_gold(gold, split):
    for name in FULL + GROW_MEMBERS + ["grow_full", "grow_full_fa"]:
        assert len(fixture(split, name)) == len(gold[split][2]), name


def test_vote_count_is_a_confidence_score(gold):
    """Precision of a link (vs sure ∪ possible) by how many of the 7 `full` members propose it (test set)."""
    members = [fixture("test", m) for m in FULL]
    hit, tot = [0] * 8, [0] * 8
    for k, (S, P) in enumerate(gold["test"][2]):
        cnt = {}
        for m in members:
            for e in m[k]:
                cnt[e] = cnt.get(e, 0) + 1
        for e, c in cnt.items():
            tot[c] += 1
            hit[c] += e in (S | P)
    prec = [hit[c] / tot[c] for c in range(1, 8)]
    for got, want in zip(prec, [0.22, 0.44, 0.68, 0.75, 0.91, 0.97, 0.996]):   # README table, 2 decimals
        assert abs(got - want) < 0.006
    assert prec == sorted(prec)                                    # monotone: more votes, higher precision


@pytest.mark.parametrize("split", ["dev", "test"])
@pytest.mark.parametrize("final_and", [False, True])
def test_grow_matches_frozen_snapshot(split, final_and):
    seed = core.vote([fixture(split, m) for m in FULL], 4)
    cand = core.vote([fixture(split, m) for m in GROW_MEMBERS], 3)
    out = [core.grow(s, c, final_and) for s, c in zip(seed, cand)]
    assert out == fixture(split, "grow_full_fa" if final_and else "grow_full")
    assert all(s <= o for s, o in zip(seed, out))                  # grow never drops a seed link


@pytest.mark.parametrize("name,ref", [("grow_full", (0.942, 0.063)), ("grow_full_fa", (0.942, 0.064))])
def test_grow_snapshot_numbers(gold, name, ref):
    r = core.evaluate(fixture("test", name), gold["test"][2])
    assert (r["F1"], r["AER"]) == ref


def test_grow_creates_multiword_blocks(gold):
    """Grow's purpose: 1-n / n-m blocks that a 1-1 vote cannot produce."""
    def multi(als):
        n = 0
        for al in als:
            src, trg = {}, {}
            for i, j in al:
                src.setdefault(i, set()).add(j)
                trg.setdefault(j, set()).add(i)
            n += sum(len(v) > 1 for v in src.values()) + sum(len(v) > 1 for v in trg.values())
        return n
    seed = core.vote([fixture("test", m) for m in FULL], 4)
    assert multi(fixture("test", "grow_full")) > multi(seed)


def test_cli_eval_on_fixture_matches_python_api(gold, capsys):
    from combalign import __main__ as cli
    from conftest import FIXTURES, GOLD
    cli.main(["eval", "--gold", f"{GOLD}/test_alignments.pharaoh", f"{FIXTURES}/test.labse8.pharaoh"])
    r = core.evaluate(fixture("test", "labse8"), gold["test"][2])
    assert f"F1 {r['F1']:.3f}" in capsys.readouterr().out
