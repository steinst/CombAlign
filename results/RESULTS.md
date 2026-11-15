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
| **3 of 7 (new, chosen on dev)** | LaBSE L8, LaBSE L6 κ.09, NLLB L12, NLLB L8 κ.09, eflomal ∩, fast_align ∩, GIZA++ ∩ | .939 | .981 | **.960** | **.043** |

## Multiword expressions: the grow ensemble

529 of the 4,424 gold blocks (connected components of gold links) are 1-n, n-1 or n-m. A block counts as *exact*
when the predicted links cover exactly the words of the block on both sides, with no link leaving the block. Phrase
extraction and dictionary induction need this property (`experiments/mwe_eval.py`).

1-1 aligners and votes cannot form such blocks. `core.grow` generalises grow-diag-final-and to an ensemble. It
starts from the 3-of-7 vote and adds neighbouring links proposed by ≥ 3 of 5 phrase-capable aligners (eflomal and
fast_align gdfa, NLLB and LaBSE IterMax, NLLB softmax).

| system | test F1 | MWE blocks exact |
|---|---|---|
| LaBSE ArgMax | .942 | 7 % |
| 3-of-7 vote | .960 | 14 % |
| eflomal gdfa | .883 | 39 % |
| **3-of-7 vote + grow** | .940 | **48 %** |

Use the vote when you need precise word links, and grow when you need phrases or dictionary entries.

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
