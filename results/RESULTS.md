# CombAlign 2.0: results on the English–Icelandic gold set

Gold set: *English-Icelandic Word Alignments 21.05* (Steingrímsson & Daníelsson, CLARIN-IS, CC BY 4.0), dev 220 and
test 384 sentence pairs with sure and possible links. Precision is measured against sure ∪ possible links, recall
against sure links, and AER as in Och & Ney (2003). Every configuration (layer, extraction method, distortion κ,
symmetrisation, training size and ensemble) is **chosen on dev and reported on test**.

The statistical aligners are trained only on ParIce 21.10 (3.45 M pairs after filtering, `experiments/prep_parice.py`)
together with the sentences being aligned. The embedding aligners need no training.

## Single aligners

| aligner | configuration chosen on dev | dev F1 | test P | test R | test F1 | test AER |
|---|---|---|---|---|---|---|
| GIZA++ (1 M ParIce pairs) | intersection | .906 | .964 | .826 | .890 | .105 |
| fast_align (all ParIce) | intersection | .910 | .914 | .864 | .888 | .109 |
| eflomal (all ParIce) | intersection | .932 | .921 | .928 | .925 | .076 |
| SimAlign mBERT | layer 8, word, ArgMax | .869 | .885 | .824 | .853 | .144 |
| SimAlign XLM-R base | layer 8, word, ArgMax, κ .09 | .894 | .930 | .880 | .904 | .093 |
| XLM-R large | layer 16, subword, ArgMax, κ .09 | .810 | .811 | .809 | .810 | .190 |
| multilingual-e5 base | layer 8, subword, ArgMax, κ .09 | .872 | .867 | .866 | .866 | .134 |
| NLLB-200 600M encoder | layer 12, word, ArgMax | .936 | .908 | .928 | .918 | .083 |
| **LaBSE** | layer 8, word, ArgMax | .954 | .928 | .957 | **.942** | .059 |

LaBSE with ArgMax at layer 6 or 8 is the best single aligner here, and it needs no training data.

## Ensembles (votes)

| ensemble | members | test P | test R | test F1 | test AER |
|---|---|---|---|---|---|
| EnsSmall 2/3 (paper) | eflomal, fast_align (gdfa), XLM-R IterMax | – | – | .934 | .070 |
| EnsLarge 3/5 (paper) | GIZA++, fast_align, eflomal (gdfa), mBERT, XLM-R ArgMax | – | – | .950 | .051 |
| **`full`: 3 of 7 (chosen on dev)** | LaBSE L8, LaBSE L6 κ.09, NLLB L12, NLLB L8 κ.09, eflomal ∩, fast_align ∩, GIZA++ ∩ | .939 | .981 | **.960** | **.043** |

## Multiword expressions: the grow ensemble

529 of the 4,424 gold blocks (connected components of gold links) are 1-n, n-1 or n-m. A block counts as *exact*
when the predicted links cover exactly the words of the block on both sides, with no link leaving the block. Phrase
extraction and dictionary induction need this property (`experiments/mwe_eval.py`).

1-1 aligners and votes rarely form such blocks. `core.grow` generalises grow-diag-final-and to an ensemble. For the
`full` recipe it starts from the 4-of-7 vote and adds neighbouring links proposed by ≥ 3 of 5 phrase-capable aligners
(eflomal and fast_align gdfa, NLLB and LaBSE IterMax, NLLB softmax). The grow settings of `emb` and `fast` are in
`combalign/recipes.py`.

| system | test F1 | test AER | MWE blocks exact |
|---|---|---|---|
| LaBSE ArgMax | .942 | .059 | 7 % |
| eflomal gdfa | .883 | .128 | 39 % |
| `emb` (2-of-3 vote) | .947 | .054 | 7 % |
| `emb --grow` | .949 | .054 | 17 % |
| `fast` (2-of-3 vote) | .954 | .046 | 7 % |
| `fast --grow` | .957 | .044 | 15 % |
| `full` (3-of-7 vote) | .960 | .043 | 14 % |
| **`full --grow`** | .942 | .063 | **44 %** |

These rows are computed from the predictions in `tests/fixtures` (`python experiments/mwe_eval.py recipes test`);
eflomal is not seeded, so a new run moves them slightly (`full --grow`: F1 .942–.943). Use the vote when you need precise word links. Use `full --grow` when
you need phrases or dictionary entries: it trades .018 F1 for three times as many exact multiword blocks. For `emb`
and `fast`, `--grow` raises both F1 and the share of exact blocks, but the share stays low.

<!-- calibration:begin (experiments/calibration.py) -->
### Confidence: precision by vote count

Share of links (vs sure ∪ possible) that are correct, by how many of the 7 `full` members proposed the link, and the
result of keeping only links with at least that many votes (`--min-votes`). Computed by `experiments/calibration.py`
from the frozen predictions.

| votes of 7 | dev links | dev precision | test links | test precision | test ≥ votes: P | R | F1 | AER |
|---|---|---|---|---|---|---|---|---|
| 1 | 398 | 0.415 | 838 | 0.215 | 0.801 | 0.996 | 0.888 | 0.125 |
| 2 | 246 | 0.634 | 432 | 0.442 | 0.897 | 0.991 | 0.942 | 0.064 |
| 3 | 181 | 0.785 | 335 | 0.684 | 0.939 | 0.981 | 0.960 | 0.043 |
| 4 | 227 | 0.824 | 425 | 0.753 | 0.959 | 0.955 | 0.957 | 0.043 |
| 5 | 247 | 0.955 | 476 | 0.908 | 0.981 | 0.913 | 0.946 | 0.052 |
| 6 | 287 | 0.962 | 714 | 0.973 | 0.991 | 0.836 | 0.907 | 0.088 |
| 7 | 1338 | 0.996 | 2741 | 0.996 | 0.996 | 0.688 | 0.814 | 0.180 |

Links added by grow (seed = vote ≥ 4/7, candidates = ≥ 3 of 5 phrase-capable aligners, final-and off), by the
number of the 5 candidate aligners that proposed them:

| candidate votes of 5 | dev links | dev precision | test links | test precision |
|---|---|---|---|---|
| 3 | 206 | 0.738 | 401 | 0.494 |
| 4 | 112 | 0.821 | 271 | 0.649 |
| 5 | 49 | 0.898 | 105 | 0.876 |
<!-- calibration:end -->

## Tokenisation (`--tokenize`)

`--tokenize` tokenises raw text (English: Moses via sacremoses; Icelandic: the Miðeind `tokenizer`; other languages:
Moses) and keeps the character offsets of every token, so links can be written as character spans
(`--output-format offsets | jsonl`). Checked with `experiments/tokenization_check.py`: the gold sentences are
detokenised, tokenised again, and compared with the gold tokens.

| split | language | sentences identical | gold tokens reproduced | new tokens that are gold tokens |
|---|---|---|---|---|
| dev | en | 214 / 220 (97.3 %) | 99.73 % | 99.23 % |
| dev | is | 214 / 220 (97.3 %) | 99.67 % | 99.37 % |
| test | en | 375 / 384 (97.7 %) | 99.84 % | 99.65 % |
| test | is | 380 / 384 (99.0 %) | 99.91 % | 99.79 % |

Conventions taken from the gold set, on top of the two tokenizers: English clitics as in the Penn Treebank
(`do n't`, `ca n't`, `it 's`, also with a typographic apostrophe: `he ’s`; Moses alone gives `don 't`); no token boundary
inside a run of letters and digits (`CYP3A4`, `m2`); `/`, `:` and `·` do not split between two digits (`2626/84`, `5:00`);
multiword tokens of the Icelandic tokenizer (dates, amounts) are split at their spaces.

Known mismatches with the gold set (all of them; the gold set is not consistent on the first three):

- numbers in parentheses: gold `(1)`, `(5)`, `(7)` as one token (6 cases), here `( 1 )`, as for every other parenthesis;
- a slash between a number and letters: gold has both `96/23/EC` (3 document numbers in each language) and `94/278 / EC`
  (5 in each language), here always `96/23 / EC`; likewise gold `EU/1/00/132/004` and `km/h`, here `EU / 1/00/132/004` and `km / h`
  (gold elsewhere: `mg / kg`, `og / eða`);
- ranges with a percent sign: gold `60-70%`, `(40-70%)`; here `60-70 %` (English) and `60 - 70%` (Icelandic);
- English: gold `Frenchman's`, `o'clock`, `y'`, `can"t`, `don't-` (a broken-off word) as one token; here `Frenchman 's`,
  `o 'clock`, `y '`, `can " t`, `don 't-`. `man. lt's` (an OCR error for "It's") keeps the full stop on `man.`;
- Icelandic: a hyphen or degree sign written onto a word stays there: `-þurrkun`, `-eða`, `20°` (gold `- þurrkun`,
  `- eða`, `20 °`).

Alignment quality on re-tokenised text. Each recipe is run on the gold tokens and on the re-tokenised sentences
(same run, statistical aligners trained on ParIce as distributed); links on the new tokens are carried back to the gold
tokens through the character offsets and scored against the gold alignments. F1 / AER:

| recipe | dev, gold tokens | dev, `--tokenize` | test, gold tokens | test, `--tokenize` |
|---|---|---|---|---|
| `emb` | .958 / .043 | .957 / .044 | .947 / .054 | .946 / .055 |
| `emb --grow` | .959 / .042 | .958 / .043 | .949 / .054 | .949 / .054 |
| `fast` | .965 / .036 | .965 / .036 | .953 / .047 | .954 / .047 |
| `fast --grow` | .965 / .036 | .965 / .036 | .957 / .044 | .957 / .044 |
| `full` | .967 / .036 | .965 / .037 | .960 / .043 | .959 / .043 |
| `full --grow` | .962 / .042 | .961 / .044 | .943 / .062 | .943 / .062 |

Tokenising raw text instead of using the gold tokens costs at most .002 F1 (dev, `full`) and .001 on test; eflomal's
sampling alone moves these numbers by about .001 between runs.

## Speed

Measured on one RTX 4090 (24 GB) in a 32-core machine shared with other jobs (load 13–20 during the runs), so the
figures are indicative. Text: the first 100,000 pairs of the ParIce training file (1.49 M English tokens).

Encoder throughput (`experiments/bench_encode.py`): sentences per second for one side of 20,000 pairs, two layers
from each forward pass, and peak GPU memory.

| model (layers) | fp16, batch 64 | fp16, batch 128 | fp32, batch 64 | fp32, batch 128 |
|---|---|---|---|---|
| LaBSE (6, 8) | 2105 (2.5 GB) | 2158 (2.5 GB) | 1715 (2.5 GB) | 1733 (3.3 GB) |
| NLLB (8, 12) | 1709 (2.0 GB) | 1761 (2.2 GB) | 1292 (3.0 GB) | 1272 (4.4 GB) |

On the CPU (fp32, batch 64, 24 threads, 2,000 sentences) the same encoders ran at 176 (LaBSE) and 89 (NLLB)
sentences per second.

End to end, `align --recipe emb` on the 100,000 pairs (loading the models, both encoders, extraction, vote, writing):

| run | wall time | pairs per second | peak RAM | peak GPU memory |
|---|---|---|---|---|
| fp16 (default), batch 64 | 3 min 01 s | 554 | 6.3 GB | 3.1 GB |
| fp16, batch 128 | 2 min 59 s | 558 | 6.3 GB | 3.1 GB |
| fp16, batch 128, `--grow` | 3 min 38 s | 459 | 6.4 GB | 3.1 GB |
| fp32, batch 64 | 4 min 07 s | 405 | 6.1 GB | 4.8 GB |

The fp16 and fp32 outputs differ in 459 of 980,447 links (0.05 %). The batch size hardly matters, because much of the
time goes to tokenisation and link extraction on the CPU. Memory does not grow with the corpus on the encoder side
(pairs are encoded in chunks of 2,000); the alignments of every member are kept in memory until the vote.

The same runs on one A100 (40 GB) in a node with two Xeon Platinum 8358 processors (64 cores), used exclusively.
On that GPU the fp32 output is identical to the frozen predictions of the `emb` recipe on the gold set, and fp16
differs in 2 of 6,742 links.

| model (layers) | fp16, batch 64 | fp16, batch 128 | fp32, batch 64 | fp32, batch 128 |
|---|---|---|---|---|
| LaBSE (6, 8) | 872 (2.5 GB) | 900 (2.5 GB) | 758 (2.5 GB) | 752 (3.3 GB) |
| NLLB (8, 12) | 794 (2.0 GB) | 816 (2.2 GB) | 586 (3.0 GB) | 583 (4.4 GB) |

On the CPU (fp32, batch 64, 2,000 sentences) the encoders ran at 339 (LaBSE) and 192 (NLLB) sentences per second
with 32 threads, 244 and 167 with 64 threads, and 224 and 155 with 16 threads.

| run | wall time | pairs per second | peak RAM | peak GPU memory |
|---|---|---|---|---|
| fp16 (default), batch 64 | 6 min 12 s | 269 | 5.5 GB | 3.1 GB |
| fp16, batch 128 | 6 min 04 s | 275 | 5.6 GB | 3.0 GB |
| fp16, batch 128, `--grow` | 6 min 22 s | 262 | 5.7 GB | 3.0 GB |
| fp32, batch 64 | 9 min 03 s | 184 | 5.5 GB | 4.5 GB |

Here the fp16 and fp32 outputs differ in 406 of 980,446 links (0.04 %).
