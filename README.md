# CombAlign

Ensemble word alignment: several aligners are run and combined by voting. The method is from

> Steinþór Steingrímsson, Hrafn Loftsson and Andy Way. 2021. *CombAlign: a Tool for Obtaining High-Quality Word
> Alignments.* In Proceedings of NoDaLiDa 2021.

Version 2.0 is a re-implementation with newer embedding aligners (LaBSE, NLLB encoder), a one-command `align` with
three recipes, confidence scores, and a `--grow` step that lets a 1-1 ensemble form multiword (1-n, n-m) alignments. On the English–Icelandic gold set the `full` recipe (3 of 7 votes) reaches
**F1 .960 / AER .043** on test; the original EnsLarge gets .950 / .051. See [results/RESULTS.md](results/RESULTS.md).

The code from the 2021 paper is in [legacy/](legacy/).

## Install

```bash
pip install -r requirements.txt     # torch, transformers, eflomal, sacremoses, tokenizer (Miðeind)
bash install.sh                     # also builds fast_align and GIZA++ into ./tools (needs cmake, a C++ compiler,
                                    # libgoogle-perftools-dev, libsparsehash-dev)
```

The `emb` recipe needs only the pip packages. `fast` needs a parallel corpus and eflomal (a pip package). `full` also
needs fast_align and GIZA++, which `install.sh` builds. Set `COMBALIGN_TOOLS` if they are built somewhere other than
`./tools`.

## Quick start: `align --recipe`

Input is one sentence per line, in two files of equal length. A line may start with an id followed by a tab, as in
the gold set. Alignments are written in Pharaoh format (`i-j`, 0-based token indices).

```bash
# no training data needed: three embedding aligners vote (2 of 3)
python -m combalign align --recipe emb --src x.en --trg x.is -o x.pharaoh

# raw text: tokenise it, write character offsets and per-link vote counts
python -m combalign align --recipe emb --tokenize --output-format jsonl --src raw.en --trg raw.is -o x.jsonl

# with a parallel corpus (fast and full need one; GIZA++ is slow, about an hour per direction on 1 M pairs)
python -m combalign align --recipe full --grow --src x.en --trg x.is \
    --train-src corpus.en --train-trg corpus.is --work work/ -o x.pharaoh --scores x.scores
```

### Recipes

The numbers are on the English–Icelandic gold set, test split; configurations were chosen on dev
(see [results/RESULTS.md](results/RESULTS.md)).

| recipe | members | vote | needs | test F1 / AER | with `--grow`, test F1 |
|---|---|---|---|---|---|
| `emb` | LaBSE L8, LaBSE L6 κ.09, NLLB L12 | 2 of 3 | nothing | .947 / .054 | .949 |
| `fast` | LaBSE L6 κ.09, NLLB L12, eflomal ∩ | 2 of 3 | a parallel corpus | .954 / .046 | .957 |
| `full` | LaBSE L8, LaBSE L6 κ.09, NLLB L12, NLLB L8 κ.09, eflomal ∩, fast_align ∩, GIZA++ ∩ | 3 of 7 | a parallel corpus, GIZA++ | .960 / .043 | .942–.943 |

`--grow` extends the vote with neighbouring links from phrase-capable aligners, so that 1-n and n-m blocks can form.
It raised F1 for `emb` and `fast`, but lowered it for `full` (see RESULTS.md), so use it with `full` only when you
need multiword links. Each recipe checks its input: `fast` and `full` without `--train-src/--train-trg` stop with a
message that points to `--recipe emb`.

With `--fp32` the encoder members reproduce the reference predictions exactly, on CPU and on GPU. The default on a
GPU is fp16, which changes a handful of links (at most 0.07 % of a voting member's links on the gold set, 0.14 % for
the softmax member that `--grow` uses; the F1 and AER of the recipes move by .001 at most). eflomal is not bit-for-bit
reproducible (its sampler is seeded internally), so its runs move F1 by about .001.

Useful options:

- `--device cuda` runs the encoders on a GPU (the default when one is available); `--batch-size` sets the encoder
  batch size (64 by default). `emb` aligns about 550 sentence pairs per second on one RTX 4090 (see RESULTS.md).
- `--work DIR` caches each member's output in `DIR`. An interrupted run resumes, and a rerun with other options only
  recomputes the members that the change affects.
- `--src-lang` / `--trg-lang` (default `en` / `is`) select the tokenizer and the NLLB language code. Two-letter codes
  are known for about 30 European languages (`combalign/simalign.py`); for any other language give the NLLB-200 code
  itself, e.g. `--trg-lang kal_Latn`. The recipes were chosen on English–Icelandic data and have not been evaluated
  on other language pairs.
- An empty line in the input gives an empty line in the output.

## Confidence

Each link can carry the number of members that proposed it, so you can choose a precision level afterwards.

```bash
python -m combalign align --recipe full --src x.en --trg x.is --train-src corpus.en --train-trg corpus.is \
    -o x.pharaoh --scores x.scores                       # Pharaoh file and a confidence file
python -m combalign align --recipe full ... --min-votes 5 -o x5.pharaoh   # keep links with ≥ 5 of 7 votes
python -m combalign filter --scores x.scores --min-votes 5 --no-grown -o x5.pharaoh   # same, without re-running
```

- `--scores FILE` writes one line per sentence, with one token per link: `i-j:c/n` (proposed by c of the n
  members), grown links `i-j:g3/5` (added by grow; 3 of the 5 candidate aligners), and `i-j:s` for a seed link whose
  count is unknown. The Pharaoh output is the same with or without `--scores`.
- `--min-votes N` keeps voted links with at least N votes. Grown links are kept unless you use `filter --no-grown`.
- `filter --scores FILE --min-votes N` builds a Pharaoh file from a scores file, so you can trade recall for
  precision without re-running the aligners.

Precision on test by the number of votes (of 7, `full`): 1 → .22, 2 → .44, 3 → .68, 4 → .75, 5 → .91, 6 → .97,
7 → .996. The full calibration table, dev and test, is in RESULTS.md.

## Tokenisation

Pass `--tokenize` to tokenise raw text. English uses Moses (sacremoses), Icelandic the Miðeind `tokenizer`, and other
languages Moses. Every token keeps its character span, so links can be mapped back to the raw text.

| `--output-format` | what is written per sentence pair |
|---|---|
| `pharaoh` (default) | `i-j` token indices |
| `offsets` | `s:e-s:e` character spans of each link, source then target |
| `jsonl` | one JSON object per pair (fields below) |

The `jsonl` object has these fields:

- `id`: the id from the input line, if there is one;
- `src`, `trg`: the raw sentences;
- `src_tokens`, `trg_tokens`: the tokens;
- `src_spans`, `trg_spans`: the `[start, end]` character span of each token (a token is always `text[start:end]`);
- `links`: a list of `{"src": i, "trg": j, "votes": c, "of": n}` (token indices; `votes` of `of` members). A grown link
  has `"grown": true` and its `votes` / `of` count the candidate aligners instead.

Offsets and jsonl also work without `--tokenize`, with the whitespace tokens.

The tokenizer also runs on its own:

```bash
python -m combalign tokenize --lang en corpus.en -o corpus.tok.en
python -m combalign tokenize --lang is corpus.is -o corpus.tok.is
```

**Tokenise a raw training corpus with `combalign tokenize` first, the same way as the input.** `--tokenize` applies
only to `--src` and `--trg`, not to `--train-src` / `--train-trg`. The reference numbers use the training corpus as
distributed; for a new corpus, tokenise it the same way as the text you align.

The tokenisation follows the gold set's conventions (details and known mismatches in RESULTS.md): English clitics as
in the Penn Treebank (`do n't`), no split inside a run of letters and digits, and no split of `/ : ·` between two
digits. Re-tokenised text costs at most .002 F1 compared with the gold tokens (see RESULTS.md).

## Advanced: the individual aligners

`align` runs these steps for you. Use the commands below to run one aligner, to try a different ensemble, or to
inspect a member. Input and output formats are the same as above.

```bash
# embedding aligners (no training data needed); --model takes a name from combalign/simalign.py
python -m combalign embed --model labse8 --src x.en --trg x.is -o labse8.pharaoh
python -m combalign embed --model labse --kappa 0.09 --src x.en --trg x.is -o labse6.pharaoh
python -m combalign embed --model nllb --src x.en --trg x.is -o nllb.pharaoh
python -m combalign embed --model nllb8 --kappa 0.09 --src x.en --trg x.is -o nllb8.pharaoh

# statistical aligners, trained on a parallel corpus plus the sentences to align
for a in eflomal fast_align giza; do
  python -m combalign stat --aligner $a --train-src corpus.en --train-trg corpus.is \
      --src x.en --trg x.is --sym intersect -o $a.pharaoh
done

# vote: keep the links proposed by at least 3 of the 7 aligners
python -m combalign vote -k 3 --scores vote.scores -o vote.pharaoh labse8.pharaoh labse6.pharaoh nllb.pharaoh \
    nllb8.pharaoh eflomal.pharaoh fast_align.pharaoh giza.pharaoh

# grow: extend the vote with links from phrase-capable aligners (multiword alignments)
python -m combalign stat --aligner eflomal --sym gdfa ... -o eflomal.gdfa.pharaoh
python -m combalign grow --seed vote.pharaoh --seed-scores vote.scores --cand-k 2 -o grow.pharaoh \
    eflomal.gdfa.pharaoh fast_align.gdfa.pharaoh nllb.itermax.pharaoh

# evaluate against a gold file (p = possible link, - = sure link)
python -m combalign eval --gold test_alignments.pharaoh vote.pharaoh grow.pharaoh
```

Other options: `embed` takes `--method argmax|itermax|match|softmax`, `--level word|bpe` and `--layer`; `stat` takes
`--sym intersect|union|gd|gdf|gdfa|fwd|rev` and `--n-train` (use only the first N training pairs). The `--model` option
also accepts a Hugging Face model id or a local directory. Model names are `mbert`, `xlmr`, `xlmrL`, `labse`, `labse8`, `labse12`, `nllb`, `nllb8` and `e5`.

From Python:

```python
from combalign import core, simalign
E = simalign.Embedder("labse8")
links = simalign.align_pair(E("I 'll have one .".split(), "en"), E("Ég þigg einn .".split(), "is"))
core.vote([aligner_a, aligner_b, aligner_c], k=2)        # lists of sets of (i, j)
core.grow(seed_links, candidate_links)
core.symmetrize(fwd, rev, "gdfa")
core.evaluate(predictions, gold)
```

### Reproducing the experiments

Put the gold set (*English-Icelandic Word Alignments 21.05*, CLARIN-IS) in `data/gold/`, then:

```bash
python experiments/prep_parice.py ParIce-train.21.10.zip              # → data/train/parice.{en,is}
python experiments/simalign_ablation.py                                # mBERT / XLM-R × level × method × κ
python experiments/emb_ablation.py                                     # LaBSE, NLLB, XLM-R large, e5
python experiments/ensemble_ablation.py align eflomal 100000,500000,1000000,all
python experiments/ensemble_ablation.py align fast_align 100000,500000,1000000,all
python experiments/ensemble_ablation.py align giza 100000,500000,1000000
python experiments/ensemble_ablation.py combine                        # → results/ensemble_{dev,test}.tsv
python experiments/grow_ensemble.py
python experiments/mwe_eval.py test                                    # → results/mwe_test.tsv
python experiments/single_table.py
python experiments/calibration.py --write                              # confidence table in RESULTS.md
python experiments/tokenization_check.py                               # tokenisation agreement with the gold set
```

`experiments/extract_phrases.py` counts phrase pairs over a whole corpus (eflomal gdfa blocks with Dice scores),
for multiword-expression and bilingual dictionary induction.

## Citation

```bibtex
@inproceedings{steingrimsson-etal-2021-combalign,
  title     = {{CombAlign}: a Tool for Obtaining High-Quality Word Alignments},
  author    = {Steingr{\'i}msson, Stein{\th}{\'o}r and Loftsson, Hrafn and Way, Andy},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  month     = may # " 31--2 " # jun,
  year      = {2021},
  address   = {Reykjavik, Iceland (Online)},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  url       = {https://aclanthology.org/2021.nodalida-main.7/},
  pages     = {64--73}
}
```

Please also cite the aligners you use: fast_align (Dyer et al. 2013), eflomal (Östling & Tiedemann 2016), GIZA++
(Och & Ney 2003), SimAlign (Jalili Sabet et al. 2020), awesome-align (Dou & Neubig 2021), LaBSE (Feng et al. 2022)
and NLLB (NLLB Team 2022).

## License

CombAlign is released under the [Apache License 2.0](LICENSE). The aligners and models it runs have their own
licences: eflomal and GIZA++ are GPL, fast_align and LaBSE are Apache 2.0, and the NLLB-200 weights are CC BY-NC 4.0
(non-commercial).
