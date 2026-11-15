# -*- coding: utf-8 -*-
"""How close is `--tokenize` to the gold set's tokenisation, and what does it cost in alignment quality?

  python experiments/tokenization_check.py                       agreement only (seconds)
  python experiments/tokenization_check.py --recipe emb          + alignment on re-tokenised text (encoders)
  python experiments/tokenization_check.py --recipe emb fast full --train-src c.en --train-trg c.is --work DIR

The gold sentences are detokenised (English: Moses detokenizer + n't joined to its verb; Icelandic: Miðeind
correct_spaces), tokenised again with combalign.tokenization, and compared with the gold tokens. With --recipe the
recipes are run on the re-tokenised sentences, the links are carried back to the gold tokens through the character
offsets, and scored against the gold alignments. --mismatches N prints the first N differing tokens per language."""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combalign import core, tokenization as tk   # noqa: E402

SPLITS = ("dev", "test")


def detokenize(toks, lang):
    if lang == "en":
        from sacremoses import MosesDetokenizer
        text = MosesDetokenizer("en").detokenize(toks)
        return re.sub(r"(?<=\w) (n['’]t)\b", r"\1", text)          # do n't → don't (Moses leaves it apart)
    import tokenizer
    return tokenizer.correct_spaces(" ".join(toks))


def prepare(gold_dir):
    """→ {split: {lang: [(text, gold spans, new spans)]}}; the gold spans locate the gold tokens in the detokenised text."""
    out = {}
    for split in SPLITS:
        src, trg, _ = core.read_gold(f"{gold_dir}/{split}")
        out[split] = {}
        for lang, sents in (("en", src), ("is", trg)):
            rows = []
            for toks in sents:
                text = detokenize(toks, lang)
                gold_spans = tk.spans_of(text, toks)
                if gold_spans is None:                               # the detokenizer changed characters: keep the gold line
                    text = " ".join(toks)
                    gold_spans = tk.whitespace_spans(text)
                rows.append((text, gold_spans, tk.tokenize(text, lang)))
            out[split][lang] = rows
    return out


def agreement(rows):
    same = sum(g == n for _, g, n in rows)
    gold = sum(len(g) for _, g, _ in rows)
    new = sum(len(n) for _, _, n in rows)
    hit = sum(len(set(g) & set(n)) for _, g, n in rows)
    return same, len(rows), hit / gold, hit / new, gold, new


def mismatches(rows):
    for text, g, n in rows:
        if g != n:
            gs, ns = set(g), set(n)
            yield (" ".join(text[a:b] for a, b in g if (a, b) not in ns), " ".join(text[a:b] for a, b in n if (a, b) not in gs), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold")
    ap.add_argument("--recipe", nargs="*", default=[], choices=("emb", "fast", "full"))
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--train-src"); ap.add_argument("--train-trg")
    ap.add_argument("--work", help="work directory; <work>/gold and <work>/retok hold the member caches")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens", nargs="*", default=["gold", "retok"], choices=("gold", "retok"))
    ap.add_argument("--mismatches", type=int, default=0)
    a = ap.parse_args()

    data = prepare(a.gold)
    print("token agreement with the gold tokenisation (gold sentences detokenised, then tokenised again)")
    print("split lang  sentences identical   gold tokens kept   new tokens in gold   tokens gold/new")
    for split in SPLITS:
        for lang in ("en", "is"):
            same, n, rec, prec, gold, new = agreement(data[split][lang])
            print(f"{split:5s} {lang}    {same:3d}/{n} ({same / n:.1%})       {rec:.2%}            {prec:.2%}            {gold}/{new}")
    if a.mismatches:
        for lang in ("en", "is"):
            print(f"\nmismatches ({lang}): gold tokens | new tokens | text")
            rows = [r for split in SPLITS for r in mismatches(data[split][lang])]
            for g, n, text in rows[:a.mismatches]:
                print(f"  {g}  |  {n}  |  {text[:70]}")
    if not a.recipe:
        return

    from combalign import pipeline
    gold = [g for split in SPLITS for g in core.read_gold(f"{a.gold}/{split}")[2]]
    n_dev = len(data["dev"]["en"])
    en = [r for split in SPLITS for r in data[split]["en"]]
    is_ = [r for split in SPLITS for r in data[split]["is"]]
    inputs = {"gold": ([tk.tokens(t, g) for t, g, _ in en], [tk.tokens(t, g) for t, g, _ in is_]),
              "retok": ([tk.tokens(t, n) for t, _, n in en], [tk.tokens(t, n) for t, _, n in is_])}
    print("\nalignment on gold tokens vs on re-tokenised text (links carried back to the gold tokens through the offsets)")
    print("recipe      tokens  dev P    R     F1    AER  | test P    R     F1    AER")
    for recipe in a.recipe:
        for grow in ([True, False] if a.grow else [False]):          # grow first: one run then computes every member
            for name, (src, trg) in inputs.items():
                if name not in a.tokens:
                    continue
                r = pipeline.run(recipe, src, trg, a.train_src, a.train_trg, grow=grow, device=a.device, fp16=False,
                                 work=f"{a.work}/{name}" if a.work else None, log=lambda m: None)
                als = r["alignments"]
                if name == "retok":
                    als = [tk.project(al, e[2], i[2], e[1], i[1]) for al, e, i in zip(als, en, is_)]
                d, t = core.evaluate(als[:n_dev], gold[:n_dev]), core.evaluate(als[n_dev:], gold[n_dev:])
                print(f"{recipe + ('+grow' if grow else ''):11s} {name:6s}  " + "  | ".join(
                    f"{x['P']:.3f} {x['R']:.3f} {x['F1']:.3f} {x['AER']:.3f}" for x in (d, t)), flush=True)


if __name__ == "__main__":
    main()
