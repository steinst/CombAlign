# -*- coding: utf-8 -*-
"""`combalign align` (recipes, caching, resume, grow, scores). CPU, fp32. Models must be in the local Hugging Face cache."""
import os

import pytest

from combalign import __main__ as cli
from combalign import core
from combalign.recipes import MEMBERS, RECIPES
from conftest import FULL, GROW_MEMBERS, GOLD, RECIPES as REF_RECIPES, fixture

os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---- recipes as data (no models, no gold) ----

def test_recipes_are_consistent_with_the_reference_definitions():
    assert set(RECIPES) == {"emb", "fast", "full"}
    for name, (members, k) in REF_RECIPES.items():
        assert RECIPES[name]["members"] == members and RECIPES[name]["k"] == k
    assert RECIPES["full"]["members"] == FULL and RECIPES["full"]["grow"]["members"] == GROW_MEMBERS
    assert RECIPES["full"]["grow"]["seed_k"] == 4 and RECIPES["full"]["grow"]["cand_k"] == 3
    for r in RECIPES.values():
        assert all(m in MEMBERS for m in r["members"] + r["grow"]["members"])
        assert r["k"] <= len(r["members"]) and r["grow"]["cand_k"] <= len(r["grow"]["members"])
        stat = any(MEMBERS[m]["kind"] == "stat" for m in r["members"])
        assert r["needs_corpus"] == stat                       # emb is the only recipe without a corpus


@pytest.mark.parametrize("recipe", ["fast", "full"])
def test_recipes_with_statistical_members_refuse_without_a_corpus_and_point_to_emb(tmp_path, recipe, capsys):
    s = tmp_path / "a.txt"
    s.write_text("a b c\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["align", "--recipe", recipe, "--src", str(s), "--trg", str(s), "-o", str(tmp_path / "o")])
    assert "--recipe emb" in str(e.value) and "--train-src" in str(e.value)


def test_unequal_input_lengths_are_refused(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("a b\nc d\n"); b.write_text("x y\n")
    with pytest.raises(SystemExit):
        cli.main(["align", "--recipe", "emb", "--src", str(a), "--trg", str(b), "-o", str(tmp_path / "o")])


# ---- end to end on 30 / 10 gold sentences ----

@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    if not os.path.exists(f"{GOLD}/test_alignments.pharaoh"):
        pytest.skip("gold set not available (data/gold)")
    src, trg, _ = core.read_gold(f"{GOLD}/test")
    d = tmp_path_factory.mktemp("align")
    for n, sents in (("s", src[:30]), ("t", trg[:30])):
        (d / n).write_text("\n".join(" ".join(x) for x in sents) + "\n", encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def emb_run(sample):
    work = sample / "work_emb"
    cli.main(["align", "--recipe", "emb", "--device", "cpu", "--src", str(sample / "s"), "--trg", str(sample / "t"),
              "--scores", str(sample / "emb.scores"), "--work", str(work), "-o", str(sample / "emb.pharaoh")])
    return work


def test_emb_recipe_equals_vote_of_frozen_members(sample, emb_run):
    members = [fixture("test", m)[:30] for m in REF_RECIPES["emb"][0]]
    assert core.read_pharaoh(sample / "emb.pharaoh") == core.vote(members, 2)


def test_emb_scores_file(sample, emb_run):
    lines = (sample / "emb.scores").read_text().splitlines()
    pharaoh = core.read_pharaoh(sample / "emb.pharaoh")
    counts = core.vote([fixture("test", m)[:30] for m in REF_RECIPES["emb"][0]], 1, return_counts=True)
    for line, al, cnt in zip(lines, pharaoh, counts):
        p = core.parse_scores(line)
        assert set(p) == al and all(n == 3 and 2 <= c == cnt[e] for e, (c, n, g) in p.items())


def test_members_are_cached_and_a_rerun_uses_only_the_cache(sample, emb_run, monkeypatch, capsys):
    assert sorted(f[:-len(".pharaoh")] for f in os.listdir(emb_run) if f.endswith(".pharaoh")) == ["labse6k09", "labse8", "nllb12"]
    from combalign import simalign
    monkeypatch.setattr(simalign, "Embedder", lambda *a, **k: pytest.fail("encoder loaded although everything is cached"))
    cli.main(["align", "--recipe", "emb", "--device", "cpu", "--src", str(sample / "s"), "--trg", str(sample / "t"),
              "--min-votes", "3", "--work", str(emb_run), "-o", str(sample / "emb3.pharaoh")])
    assert capsys.readouterr().err.count("[cache]") == 3
    assert core.read_pharaoh(sample / "emb3.pharaoh") == core.vote([fixture("test", m)[:30] for m in REF_RECIPES["emb"][0]], 3)


def test_resume_recomputes_only_the_missing_member(sample, emb_run, capsys):
    os.remove(emb_run / "nllb12.pharaoh")
    cli.main(["align", "--recipe", "emb", "--device", "cpu", "--src", str(sample / "s"), "--trg", str(sample / "t"),
              "--work", str(emb_run), "-o", str(sample / "emb_resume.pharaoh")])
    err = capsys.readouterr().err
    assert err.count("[cache]") == 2 and "[done] nllb12" in err and "[done] labse8" not in err
    assert core.read_pharaoh(sample / "emb_resume.pharaoh") == core.read_pharaoh(sample / "emb.pharaoh")


def test_changed_input_invalidates_the_cache(sample, emb_run, capsys):
    s2 = sample / "s2"
    lines = (sample / "s").read_text().splitlines()
    s2.write_text("\n".join(lines[:-1] + [lines[-1] + " extra"]) + "\n", encoding="utf-8")
    cli.main(["align", "--recipe", "emb", "--device", "cpu", "--src", str(s2), "--trg", str(sample / "t"),
              "--work", str(emb_run), "-o", str(sample / "emb_changed.pharaoh")])
    assert "[cache]" not in capsys.readouterr().err
    # restore the cache for the tests that follow
    cli.main(["align", "--recipe", "emb", "--device", "cpu", "--src", str(sample / "s"), "--trg", str(sample / "t"),
              "--work", str(emb_run), "-o", str(sample / "emb.pharaoh")])


def test_emb_grow_equals_grow_of_frozen_members(sample, emb_run):
    cli.main(["align", "--recipe", "emb", "--grow", "--device", "cpu", "--src", str(sample / "s"), "--trg", str(sample / "t"),
              "--scores", str(sample / "g.scores"), "--work", str(emb_run), "-o", str(sample / "g.pharaoh")])
    P = lambda n: fixture("test", n)[:30]   # noqa: E731
    seed = core.vote([P(m) for m in REF_RECIPES["emb"][0]], 2)
    cand = core.vote([P(m) for m in RECIPES["emb"]["grow"]["members"]], 3)
    want = [core.grow(s, c, False) for s, c in zip(seed, cand)]
    got = core.read_pharaoh(sample / "g.pharaoh")
    assert got == want
    for line, al, s in zip((sample / "g.scores").read_text().splitlines(), got, seed):
        p = core.parse_scores(line)
        assert set(p) == al and {e for e, v in p.items() if not v[2]} == s
        assert all(v[1] == 3 for v in p.values())              # 3 voters / 3 grow members


def test_fast_recipe_runs_end_to_end_on_a_tiny_corpus(sample, tmp_path, capsys):
    """Plumbing only (a 200-pair 'corpus' gives poor eflomal alignments): stat member trained in a thread, cached,
    voted with the two encoders."""
    if not os.path.exists(f"{GOLD}/dev_alignments.pharaoh"):
        pytest.skip("gold set not available (data/gold)")
    try:
        import eflomal  # noqa: F401
    except ImportError:
        pytest.skip("eflomal not installed")
    ds, dt, _ = core.read_gold(f"{GOLD}/dev")
    (tmp_path / "c.en").write_text("\n".join(" ".join(x) for x in ds) + "\n", encoding="utf-8")
    (tmp_path / "c.is").write_text("\n".join(" ".join(x) for x in dt) + "\n", encoding="utf-8")
    work = tmp_path / "w"
    args = ["align", "--recipe", "fast", "--device", "cpu", "--src", str(sample / "s"), "--trg", str(sample / "t"),
            "--train-src", str(tmp_path / "c.en"), "--train-trg", str(tmp_path / "c.is"), "--work", str(work),
            "-o", str(tmp_path / "o")]
    cli.main(args)
    assert len(core.read_pharaoh(tmp_path / "o")) == 30
    assert os.path.exists(work / "eflomal.pharaoh") and os.path.exists(work / "labse6k09.pharaoh")
    ef = core.read_pharaoh(work / "eflomal.pharaoh")
    assert core.read_pharaoh(tmp_path / "o") == core.vote([fixture("test", "labse6k09")[:30], fixture("test", "nllb12")[:30], ef], 2)
    # a different training file invalidates only the statistical member
    (tmp_path / "c.en").write_text("a b c\n" + (tmp_path / "c.en").read_text(), encoding="utf-8")
    (tmp_path / "c.is").write_text("x y z\n" + (tmp_path / "c.is").read_text(), encoding="utf-8")
    capsys.readouterr()
    cli.main(args)
    err = capsys.readouterr().err
    assert "[start] eflomal" in err and err.count("[cache]") == 2          # only the statistical member is recomputed


@pytest.mark.slow
@pytest.mark.parametrize("split", ["dev", "test"])
def test_emb_recipe_on_the_full_gold_sets_reproduces_the_reference(tmp_path, split):
    if not os.path.exists(f"{GOLD}/{split}_alignments.pharaoh"):
        pytest.skip("gold set not available (data/gold)")
    cli.main(["align", "--recipe", "emb", "--device", "cpu", "--fp32", "--src", f"{GOLD}/{split}_sentences.en",
              "--trg", f"{GOLD}/{split}_sentences.is", "--work", str(tmp_path / "w"), "-o", str(tmp_path / "o")])
    out = core.read_pharaoh(tmp_path / "o")
    assert out == core.vote([fixture(split, m) for m in REF_RECIPES["emb"][0]], 2)
    r = core.evaluate(out, core.read_gold(f"{GOLD}/{split}")[2])
    assert (r["F1"], r["AER"]) == {"dev": (0.958, 0.043), "test": (0.947, 0.054)}[split]
