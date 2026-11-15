# -*- coding: utf-8 -*-
"""Confidence output: vote counts, scores files, --min-votes, grown-link marks."""
import pytest

from combalign import __main__ as cli
from combalign import core
from conftest import FIXTURES, FULL, GROW_MEMBERS, fixture


def test_vote_return_counts_hand_made():
    A, B, C = [{(0, 0), (1, 1)}], [{(0, 0)}], [{(0, 0), (2, 2)}]
    assert core.vote([A, B, C], 1, return_counts=True) == [{(0, 0): 3, (1, 1): 1, (2, 2): 1}]
    assert core.vote([A, B, C], 2, return_counts=True) == [{(0, 0): 3}]
    assert core.vote([A, B, C], 2) == [{(0, 0)}]                          # default unchanged: plain sets
    assert isinstance(core.vote([A, B, C], 2)[0], set)


def test_grow_return_added():
    al, added = core.grow({(0, 0)}, {(1, 1), (5, 5)}, final_and=True, return_added=True)
    assert al == {(0, 0), (1, 1), (5, 5)} and added == {(1, 1), (5, 5)}
    assert core.grow({(0, 0)}, {(1, 1)}, True) == {(0, 0), (1, 1)}         # default unchanged


def test_scores_format_roundtrip():
    line = core.to_scores({(0, 0): 7, (3, 2): 4}, 7, {(1, 1), (4, 3)}, {(1, 1): 3, (4, 3): 5}, 5)
    assert line == "0-0:7/7 1-1:g3/5 3-2:4/7 4-3:g5/5"                    # sorted by position; one token per link
    p = core.parse_scores(line)
    assert p == {(0, 0): (7, 7, False), (3, 2): (4, 7, False), (1, 1): (3, 5, True), (4, 3): (5, 5, True)}
    assert core.parse_scores("") == {}
    assert core.to_scores({(0, 0): None}, None) == "0-0:s" and core.parse_scores("0-0:s") == {(0, 0): (None, None, False)}
    with pytest.raises(ValueError):
        core.parse_scores("0-0:x/7")


def test_filter_scores():
    p = core.parse_scores("0-0:7/7 1-1:g3/5 3-2:4/7 5-5:s")
    assert core.filter_scores(p, 5) == {(0, 0), (1, 1), (5, 5)}            # voted links ≥ 5, grown kept, uncounted kept
    assert core.filter_scores(p, 5, keep_grown=False) == {(0, 0), (5, 5)}
    assert core.filter_scores(p, 1) == set(p)


def _w(p, lines):
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_cli_vote_scores_and_min_votes_hand_made(tmp_path):
    a = _w(tmp_path / "a", ["0-0 1-1", "0-0"]); b = _w(tmp_path / "b", ["0-0", "0-0 2-2"]); c = _w(tmp_path / "c", ["0-0 2-2", "2-2"])
    out, sc = tmp_path / "o", tmp_path / "s"
    cli.main(["vote", "-k", "1", "--scores", str(sc), "-o", str(out), a, b, c])
    assert sc.read_text() == "0-0:3/3 1-1:1/3 2-2:1/3\n0-0:2/3 2-2:2/3\n"
    assert out.read_text() == "0-0 1-1 2-2\n0-0 2-2\n"                      # Pharaoh output unchanged by --scores
    cli.main(["vote", "-k", "1", "--min-votes", "2", "--scores", str(sc), "-o", str(out), a, b, c])
    assert out.read_text() == "0-0\n0-0 2-2\n" and sc.read_text() == "0-0:3/3\n0-0:2/3 2-2:2/3\n"


def test_cli_grow_scores_hand_made(tmp_path):
    seed = _w(tmp_path / "seed", ["0-0"]); a = _w(tmp_path / "a", ["0-0 1-1 5-5"]); b = _w(tmp_path / "b", ["1-1 5-5"])
    ss = _w(tmp_path / "ss", ["0-0:6/7"])
    out, sc = tmp_path / "o", tmp_path / "s"
    cli.main(["grow", "--seed", seed, "--seed-scores", ss, "--cand-k", "2", "--scores", str(sc), "-o", str(out), a, b])
    assert out.read_text() == "0-0 1-1 5-5\n"
    assert sc.read_text() == "0-0:6/7 1-1:g2/2 5-5:g2/2\n"
    cli.main(["grow", "--seed", seed, "--cand-k", "2", "--scores", str(sc), "-o", str(out), a, b])    # no seed counts
    assert sc.read_text() == "0-0:s 1-1:g2/2 5-5:g2/2\n"
    cli.main(["filter", "--scores", str(sc), "--no-grown", "-o", str(out)])
    assert out.read_text() == "0-0\n"
    with pytest.raises(SystemExit):
        cli.main(["grow", "--seed", seed, "--cand-k", "2", "--min-votes", "3", "-o", str(out), a, b])


# ---- on the frozen predictions of the `full` recipe (no gold needed) ----

@pytest.fixture(scope="module")
def full_scores(tmp_path_factory):
    d = tmp_path_factory.mktemp("sc")
    members = [f"{FIXTURES}/test.{m}.pharaoh" for m in FULL]
    cli.main(["vote", "-k", "1", "--scores", str(d / "all.scores"), "-o", str(d / "all.pharaoh")] + members)
    return d, members


def test_scores_file_matches_pharaoh_and_counts(full_scores):
    d, members = full_scores
    lines = (d / "all.scores").read_text().splitlines()
    pharaoh = core.read_pharaoh(d / "all.pharaoh")
    counts = core.vote([fixture("test", m) for m in FULL], 1, return_counts=True)
    assert len(lines) == len(pharaoh) == len(counts)
    for line, al, cnt in zip(lines, pharaoh, counts):
        p = core.parse_scores(line)
        assert set(p) == al == set(cnt) and all(n == 7 and c == cnt[e] for e, (c, n, g) in p.items())


@pytest.mark.parametrize("k", [2, 3, 4, 7])
def test_filter_min_votes_equals_vote_at_higher_k(full_scores, k, tmp_path):
    d, members = full_scores
    cli.main(["filter", "--scores", str(d / "all.scores"), "--min-votes", str(k), "-o", str(tmp_path / "f")])
    assert core.read_pharaoh(tmp_path / "f") == core.vote([fixture("test", m) for m in FULL], k)


def test_grow_scores_on_fixtures_mark_exactly_the_grown_links(tmp_path):
    seed_p, seed_s = tmp_path / "seed.pharaoh", tmp_path / "seed.scores"
    cli.main(["vote", "-k", "4", "--scores", str(seed_s), "-o", str(seed_p)] + [f"{FIXTURES}/test.{m}.pharaoh" for m in FULL])
    out, sc = tmp_path / "g.pharaoh", tmp_path / "g.scores"
    cli.main(["grow", "--seed", str(seed_p), "--seed-scores", str(seed_s), "--cand-k", "3", "--no-final-and", "--scores", str(sc),
              "-o", str(out)] + [f"{FIXTURES}/test.{m}.pharaoh" for m in GROW_MEMBERS])
    assert core.read_pharaoh(out) == fixture("test", "grow_full")            # Pharaoh output == frozen snapshot
    seed = core.read_pharaoh(seed_p)
    for line, al, s in zip(sc.read_text().splitlines(), core.read_pharaoh(out), seed):
        p = core.parse_scores(line)
        assert set(p) == al
        assert {e for e, v in p.items() if not v[2]} == s                    # non-grown = seed, grown = the rest
        assert all(v[0] >= 4 and v[1] == 7 for e, v in p.items() if not v[2])
        assert all(3 <= v[0] <= 5 and v[1] == 5 for e, v in p.items() if v[2])


def test_calibration_precision_rises_with_votes():
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(f"{root}/data/gold/test_alignments.pharaoh"):
        pytest.skip("gold set not available (data/gold)")
    spec = importlib.util.spec_from_file_location("calibration", f"{root}/experiments/calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = mod.tables(FIXTURES)
    for split in ("dev", "test"):
        prec = [h / n for h, n in (t[split][0][c] for c in range(1, 8))]
        assert all(b >= a - 0.01 for a, b in zip(prec, prec[1:]))          # more votes, higher precision (dev 5→6 dips .007)
        assert prec[0] < 0.5 < prec[-1] and prec[-1] > 0.99
