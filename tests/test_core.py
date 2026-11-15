# -*- coding: utf-8 -*-
"""Core logic on small hand-made cases (no data, no models)."""
import pytest

from combalign import __main__ as cli
from combalign import core


def test_parse_pharaoh_sure_and_possible():
    sure, poss = core.parse_pharaoh("0-0 1p1 12-3 4p10")
    assert sure == {(0, 0), (12, 3)}
    assert poss == {(1, 1), (4, 10)}
    assert core.parse_pharaoh("") == (set(), set())


def test_pharaoh_roundtrip_and_ordering(tmp_path):
    als = [{(2, 1), (0, 0)}, set(), {(10, 2), (1, 1)}]
    assert core.to_pharaoh(als[0]) == "0-0 2-1"
    p = tmp_path / "a.pharaoh"
    core.write_pharaoh(p, als)
    assert p.read_text().split("\n") == ["0-0 2-1", "", "1-1 10-2", ""]
    assert core.read_pharaoh(p) == als


def test_read_pharaoh_merges_possible_into_links(tmp_path):
    p = tmp_path / "g.pharaoh"
    p.write_text("0-0 1p1\n")
    assert core.read_pharaoh(p) == [{(0, 0), (1, 1)}]


def test_read_lines_with_and_without_ids(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("1\ta b c\nd e\n", encoding="utf-8")
    assert core.read_lines(p) == [["a", "b", "c"], ["d", "e"]]


def test_evaluate_hand_computed():
    gold = [({(0, 0), (1, 1)}, {(2, 2)})]
    r = core.evaluate([{(0, 0), (2, 2), (3, 3)}], gold)
    # 3 predicted, 2 correct vs S∪P, 1 of 2 sure found: P .667, R .5, AER 1 - (1+2)/(3+2) = .4
    assert (r["P"], r["R"], r["F1"], r["AER"], r["edges"]) == (0.667, 0.5, 0.571, 0.4, 3)


def test_evaluate_perfect_and_empty():
    gold = [({(0, 0)}, {(1, 0)})]
    assert core.evaluate([{(0, 0)}], gold)["AER"] == 0.0
    r = core.evaluate([set()], gold)
    assert (r["P"], r["R"], r["F1"], r["edges"]) == (0.0, 0.0, 0.0, 0)


def test_symmetrize_intersect_union():
    fwd, rev = {(0, 0), (1, 1)}, {(1, 1), (2, 2)}
    assert core.symmetrize(fwd, rev, "intersect") == {(1, 1)}
    assert core.symmetrize(fwd, rev, "union") == {(0, 0), (1, 1), (2, 2)}


def test_symmetrize_grow_diag_variants():
    # (5,5) / (5,6) are not neighbours of the intersection {(0,0)}: grow-diag leaves them out, the "final"
    # steps differ: gdf adds links with an unaligned source OR target, gdfa only if BOTH are unaligned.
    fwd, rev = {(0, 0), (5, 5)}, {(0, 0), (5, 6)}
    assert core.symmetrize(fwd, rev, "gd") == {(0, 0)}
    assert core.symmetrize(fwd, rev, "gdf") == {(0, 0), (5, 5), (5, 6)}
    assert core.symmetrize(fwd, rev, "gdfa") == {(0, 0), (5, 5)}


def test_symmetrize_grow_diag_adds_neighbours():
    fwd, rev = {(0, 0), (1, 0)}, {(0, 0), (1, 1)}
    assert core.symmetrize(fwd, rev, "gd") == {(0, 0), (1, 0), (1, 1)}


@pytest.mark.parametrize("how", ["intersect", "gd", "gdf", "gdfa"])
def test_symmetrize_contains_intersection_and_inside_union(how):
    fwd = {(0, 0), (1, 1), (2, 1), (4, 3)}
    rev = {(0, 0), (1, 2), (2, 1), (4, 4)}
    out = core.symmetrize(fwd, rev, how)
    assert fwd & rev <= out <= fwd | rev


def test_vote_thresholds():
    A = [{(0, 0), (1, 1)}, {(0, 0)}]
    B = [{(0, 0)}, {(1, 1)}]
    C = [{(0, 0), (2, 2)}, {(1, 1), (0, 0)}]
    assert core.vote([A, B, C], 1) == [{(0, 0), (1, 1), (2, 2)}, {(0, 0), (1, 1)}]
    assert core.vote([A, B, C], 2) == [{(0, 0)}, {(0, 0), (1, 1)}]
    assert core.vote([A, B, C], 3) == [{(0, 0)}, set()]
    assert core.vote([A, B, C], 4) == [set(), set()]


def test_vote_k1_is_union_and_kn_is_intersection():
    A, B = [{(0, 0), (1, 1)}], [{(1, 1), (2, 2)}]
    assert core.vote([A, B], 1) == [{(0, 0), (1, 1), (2, 2)}]
    assert core.vote([A, B], 2) == [{(1, 1)}]


def test_grow_adds_neighbours_only_when_a_word_is_free():
    seed = {(0, 0)}
    cand = {(0, 0), (1, 1), (2, 2), (5, 5)}
    # (1,1) diagonal neighbour of the seed, then (2,2) neighbour of (1,1); (5,5) touches nothing
    assert core.grow(seed, cand, final_and=False) == {(0, 0), (1, 1), (2, 2)}
    # final-and picks up (5,5): both its words are still unaligned
    assert core.grow(seed, cand, final_and=True) == {(0, 0), (1, 1), (2, 2), (5, 5)}


def test_grow_skips_links_whose_words_are_both_aligned():
    assert core.grow({(0, 0), (1, 1)}, {(0, 1)}, final_and=True) == {(0, 0), (1, 1)}


def test_grow_keeps_seed_and_forms_blocks():
    seed = {(0, 0), (3, 3)}
    cand = {(0, 1), (1, 0), (3, 4)}              # 1-n / n-1 around (0,0), 1-n at (3,3)
    out = core.grow(seed, cand, final_and=False)
    assert seed <= out and out == {(0, 0), (0, 1), (1, 0), (3, 3), (3, 4)}


def test_grow_empty_seed_final_and_only():
    assert core.grow(set(), {(0, 0), (0, 1)}, final_and=False) == set()
    assert core.grow(set(), {(0, 0), (1, 1)}, final_and=True) == {(0, 0), (1, 1)}


# ---- command line (vote / grow / eval work on Pharaoh files only) ----

def _write(p, lines):
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_cli_vote(tmp_path):
    a = _write(tmp_path / "a", ["0-0 1-1", "0-0"])
    b = _write(tmp_path / "b", ["0-0", "0-0 2-2"])
    c = _write(tmp_path / "c", ["0-0 2-2", "2-2"])
    out = tmp_path / "v"
    cli.main(["vote", "-k", "2", "-o", str(out), a, b, c])
    assert out.read_text().split("\n") == ["0-0", "0-0 2-2", ""]


def test_cli_grow(tmp_path):
    seed = _write(tmp_path / "seed", ["0-0"])
    a = _write(tmp_path / "a", ["0-0 1-1 5-5"])
    b = _write(tmp_path / "b", ["1-1 5-5"])
    out = tmp_path / "g"
    cli.main(["grow", "--seed", seed, "--cand-k", "2", "-o", str(out), a, b])
    assert out.read_text() == "0-0 1-1 5-5\n"          # candidates 1-1, 5-5 (≥ 2 votes); 5-5 via final-and
    cli.main(["grow", "--seed", seed, "--cand-k", "2", "--no-final-and", "-o", str(out), a, b])
    assert out.read_text() == "0-0 1-1\n"


def test_cli_eval(tmp_path, capsys):
    gold = _write(tmp_path / "gold", ["0-0 1-1 2p2"])
    pred = _write(tmp_path / "pred", ["0-0 2-2 3-3"])
    cli.main(["eval", "--gold", gold, pred])
    line = capsys.readouterr().out
    assert "P 0.667" in line and "R 0.500" in line and "F1 0.571" in line and "AER 0.400" in line
