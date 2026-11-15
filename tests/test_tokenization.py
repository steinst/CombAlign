# -*- coding: utf-8 -*-
"""Tokenisation with character offsets, projection between tokenisations, and --output-format offsets | jsonl."""
import json
import os

import pytest

from combalign import __main__ as cli
from combalign import core
from combalign import tokenization as tk
from conftest import GOLD, ROOT, RECIPES as REF_RECIPES, fixture

os.environ.setdefault("HF_HUB_OFFLINE", "1")
pytest.importorskip("sacremoses")


def toks(text, lang):
    spans = tk.tokenize(text, lang)
    assert all(0 <= a < b <= len(text) for a, b in spans)                           # tokens are substrings of the text,
    assert all(x[1] <= y[0] for x, y in zip(spans, spans[1:]))                      # in order, not overlapping,
    assert "".join(text[a:b] for a, b in spans) == "".join(text.split())            # and nothing but whitespace is lost
    return tk.tokens(text, spans)


# ---- English (Moses + the gold set's conventions) ----

@pytest.mark.parametrize("text, expected", [
    ("I'll have one of those.", "I 'll have one of those ."),
    ("I don't know, she can't and won't.", "I do n't know , she ca n't and wo n't ."),
    ("It’s late and he couldn’t come.", "It ’s late and he could n’t come ."),
    ("He isn't here; they're gone.", "He is n't here ; they 're gone ."),
    ("Regulation (EEC) No 2626/84 at 5:00, i.e. 1,500.5 units.", "Regulation ( EEC ) No 2626/84 at 5:00 , i.e. 1,500.5 units ."),
    ("mg/kg and/or 96/23/EC", "mg / kg and / or 96/23 / EC"),
    ("  two  spaces\tand a tab  ", "two spaces and a tab"),
    ("AT&T <b> &amp; \"quotes\"", "AT & T < b > & amp ; \" quotes \""),          # no HTML escaping of the tokens
    ("", ""),
])
def test_english(text, expected):
    assert toks(text, "en") == expected.split()


def test_offsets_point_into_the_raw_text():
    text = "  I don't  think it’s 5:00."
    spans = tk.tokenize(text, "en")
    assert tk.tokens(text, spans) == ["I", "do", "n't", "think", "it", "’s", "5:00", "."]
    assert spans[:3] == [(2, 3), (4, 6), (6, 9)] and text[spans[5][0]:spans[5][1]] == "’s"


def test_other_languages_use_moses_with_their_own_rules():
    assert toks("Das ist z.B. ein Test, d.h. es klappt um 14:30 Uhr.", "de") == \
        "Das ist z.B. ein Test , d.h. es klappt um 14:30 Uhr .".split()
    assert toks("I don't", "de") != ["I", "do", "n't"]                              # the English clitic rule is English only


# ---- Icelandic (Miðeind tokenizer) ----

@pytest.mark.parametrize("text, expected", [
    ("Hann kom 3. janúar 2020 kl. 14:30 og borgaði 1.500 kr. fyrir þetta, þ.e.a.s. 10% afslátt.",
     "Hann kom 3. janúar 2020 kl. 14:30 og borgaði 1.500 kr. fyrir þetta , þ.e.a.s. 10% afslátt ."),   # multiword tokens split
    ("„Halló,“ sagði hún. Þetta er t.d. gott (eða hvað?) – já!",
     "„ Halló , “ sagði hún . Þetta er t.d. gott ( eða hvað ? ) – já !"),                                # two sentences, one line
    ("Sjá 5. gr. laga nr. 2626/84 og CYP3A4 hemla a.m.k. 1250 mg/m2.", "Sjá 5. gr. laga nr. 2626/84 og CYP3A4 hemla a.m.k. 1250 mg / m2 ."),
    ("  Póstur: jon@example.is &amp; ½ bolli… ", "Póstur : jon@example.is & amp ; ½ bolli …"),
    ("", ""),
])
def test_icelandic(text, expected):
    pytest.importorskip("tokenizer")
    assert toks(text, "is") == expected.split()


def test_unmappable_tokenizer_output_falls_back_to_whitespace_tokens(monkeypatch, capsys):
    monkeypatch.setattr(tk, "_moses_spans", lambda text, lang: None)
    assert tk.tokenize("a b, c", "en") == [(0, 1), (2, 4), (5, 6)]
    assert "whitespace tokens" in capsys.readouterr().err
    assert tk.spans_of("a b, c", ["a", "b", ";", "c"]) is None and tk.spans_of("a b,", ["a", "b", ","]) == [(0, 1), (2, 3), (3, 4)]
    assert tk.spans_of("a b", ["a"]) is None                                         # text left over


# ---- projection and output formats ----

def test_project_carries_links_to_every_overlapping_token():
    #         0123456789
    src_a, src_b = [(0, 5), (6, 9)], [(0, 2), (2, 5), (6, 9)]                        # "don't yes" as don't|yes and do|n't|yes
    trg = [(0, 3), (4, 7)]
    assert tk.project({(0, 1), (1, 0)}, src_a, trg, src_b, trg) == {(0, 1), (1, 1), (2, 0)}
    assert tk.project({(0, 1), (1, 1), (2, 0)}, src_b, trg, src_a, trg) == {(0, 1), (1, 0)}
    assert tk.project({(0, 0)}, src_a, trg, src_a, trg) == {(0, 0)}


def test_offsets_and_jsonl_lines():
    src, trg = "I don't.", "Ég geri það ekki."
    ss, ts = tk.tokenize(src, "en"), tk.whitespace_spans(trg)
    links = {(0, 0), (1, 1), (2, 3)}
    assert tk.to_offsets(links, ss, ts) == "0:1-0:2 2:4-3:7 4:7-12:17"
    d = json.loads(tk.to_jsonl(links, src, trg, ss, ts, sid="7", votes={(0, 0): 3, (1, 1): 2}, n=3, grown={(2, 3)},
                               grown_counts={(2, 3): 4}, m=5))
    assert d["id"] == "7" and d["src"] == src and d["src_tokens"] == ["I", "do", "n't", "."] and d["trg_spans"][3] == [12, 17]
    assert d["links"] == [{"src": 0, "trg": 0, "votes": 3, "of": 3}, {"src": 1, "trg": 1, "votes": 2, "of": 3},
                          {"src": 2, "trg": 3, "votes": 4, "of": 5, "grown": True}]
    plain = json.loads(tk.to_jsonl(links, src, trg, ss, ts))
    assert "id" not in plain and plain["links"][0] == {"src": 0, "trg": 0}


def test_tokenize_command_keeps_ids(tmp_path):
    (tmp_path / "raw").write_text("1\tI don't know.\n2\tIt's 5:00.\n", encoding="utf-8")
    cli.main(["tokenize", "--lang", "en", str(tmp_path / "raw"), "-o", str(tmp_path / "tok")])
    assert (tmp_path / "tok").read_text(encoding="utf-8") == "1\tI do n't know .\n2\tIt 's 5:00 .\n"
    assert core.read_lines(tmp_path / "tok")[0] == ["I", "do", "n't", "know", "."]


# ---- through `align` (CPU, emb recipe, gold sentences) ----

@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    pytest.importorskip("tokenizer")
    if not os.path.exists(f"{GOLD}/test_alignments.pharaoh"):
        pytest.skip("gold set not available (data/gold)")
    d = tmp_path_factory.mktemp("tok")
    n = 30
    for lang in ("en", "is"):
        lines = open(f"{GOLD}/test_sentences.{lang}", encoding="utf-8").read().split("\n")[:n]
        (d / f"tok.{lang}").write_text("\n".join(lines) + "\n", encoding="utf-8")           # tokenised, with ids
    # raw text for the same sentences: the first 4 by hand (others would do), written as a user would have them
    raw_en = ["Let me try.", "I don't know.", "It's 5:00, isn't it?", "Regulation (EEC) No 2626/84."]
    raw_is = ["Leyfðu mér að reyna.", "Ég veit það ekki.", "Klukkan er 5:00, er það ekki?", "Reglugerð (EBE) nr. 2626/84."]
    (d / "raw.en").write_text("\n".join(raw_en) + "\n", encoding="utf-8")
    (d / "raw.is").write_text("\n".join(raw_is) + "\n", encoding="utf-8")
    return d


def test_output_formats_without_tokenize_describe_the_same_alignment(sample):
    out = {}
    for fmt in ("pharaoh", "offsets", "jsonl"):
        out[fmt] = sample / f"emb.{fmt}"
        cli.main(["align", "--recipe", "emb", "--device", "cpu", "--src", str(sample / "tok.en"), "--trg", str(sample / "tok.is"),
                  "--work", str(sample / "work"), "--output-format", fmt, "--scores", str(sample / f"emb.{fmt}.scores"),
                  "-o", str(out[fmt])])
    pharaoh = core.read_pharaoh(out["pharaoh"])
    assert pharaoh == core.vote([fixture("test", m)[:30] for m in REF_RECIPES["emb"][0]], 2)
    ids, src = core.read_raw(sample / "tok.en")
    _, trg = core.read_raw(sample / "tok.is")
    assert ids[0] == "1" and not src[0].startswith("1")
    scores = [core.parse_scores(l) for l in open(sample / "emb.jsonl.scores", encoding="utf-8")]
    for k, (line, off) in enumerate(zip(open(out["jsonl"], encoding="utf-8"), open(out["offsets"], encoding="utf-8"))):
        d = json.loads(line)
        assert d["id"] == ids[k] and d["src"] == src[k] and d["trg"] == trg[k]
        assert d["src_tokens"] == src[k].split() and d["trg_tokens"] == trg[k].split()
        assert [d["src"][a:b] for a, b in d["src_spans"]] == d["src_tokens"]
        assert {(l["src"], l["trg"]) for l in d["links"]} == pharaoh[k]
        assert all(l["of"] == 3 and scores[k][(l["src"], l["trg"])][0] == l["votes"] for l in d["links"])
        assert off.split() == [f"{d['src_spans'][i][0]}:{d['src_spans'][i][1]}-{d['trg_spans'][j][0]}:{d['trg_spans'][j][1]}"
                               for i, j in sorted(pharaoh[k])]


def test_align_with_tokenize_equals_align_on_the_tokenised_text(sample):
    cli.main(["align", "--recipe", "emb", "--grow", "--device", "cpu", "--tokenize", "--src", str(sample / "raw.en"),
              "--trg", str(sample / "raw.is"), "--output-format", "jsonl", "-o", str(sample / "raw.jsonl")])
    for lang in ("en", "is"):
        cli.main(["tokenize", "--lang", lang, str(sample / f"raw.{lang}"), "-o", str(sample / f"rawtok.{lang}")])
    cli.main(["align", "--recipe", "emb", "--grow", "--device", "cpu", "--src", str(sample / "rawtok.en"),
              "--trg", str(sample / "rawtok.is"), "-o", str(sample / "rawtok.pharaoh")])
    ref = core.read_pharaoh(sample / "rawtok.pharaoh")
    recs = [json.loads(l) for l in open(sample / "raw.jsonl", encoding="utf-8")]
    assert recs[1]["src_tokens"] == ["I", "do", "n't", "know", "."] and recs[3]["trg_tokens"] == ["Reglugerð", "(", "EBE", ")", "nr.", "2626/84", "."]
    assert recs[1]["src"] == "I don't know." and "id" not in recs[1]
    for d, al in zip(recs, ref):
        assert {(l["src"], l["trg"]) for l in d["links"]} == al and len(al) >= 3
        assert [d["src"][a:b] for a, b in d["src_spans"]] == d["src_tokens"]
        assert [d["trg"][a:b] for a, b in d["trg_spans"]] == d["trg_tokens"]
        assert all(("votes" in l) for l in d["links"])


def test_gold_agreement_does_not_regress():
    """Detokenise the gold sentences, tokenise again: share of gold tokens reproduced (see results/RESULTS.md)."""
    pytest.importorskip("tokenizer")
    if not os.path.exists(f"{GOLD}/test_alignments.pharaoh"):
        pytest.skip("gold set not available (data/gold)")
    import sys
    sys.path.insert(0, f"{ROOT}/experiments")
    import tokenization_check as tc
    data = tc.prepare(GOLD)
    for split in ("dev", "test"):
        for lang in ("en", "is"):
            same, n, kept, _, _, _ = tc.agreement(data[split][lang])
            assert kept >= 0.995 and same / n >= 0.96, (split, lang, same, n, kept)
