# CombAlign

Ensemble word alignment: several aligners are run and combined by voting. The method is from

> Steinþór Steingrímsson, Hrafn Loftsson and Andy Way. 2021. *CombAlign: a Tool for Obtaining High-Quality Word
> Alignments.* In Proceedings of NoDaLiDa 2021.

Version 2.0 is a re-implementation with newer embedding aligners (LaBSE, NLLB encoder, awesome-align fine-tuning)
and a **grow** step that lets a 1-1 ensemble form multiword (1-n, n-m) alignments. On the English–Icelandic gold set
a 3-of-7 vote reaches **F1 .960 / AER .043** on test; the original EnsLarge gets .950 / .051. See
[results/RESULTS.md](results/RESULTS.md).

The code from the 2021 paper is in [legacy/](legacy/).

## Install

```bash
pip install -r requirements.txt     # torch, transformers, eflomal, awesome-align, scipy
bash install.sh                     # also builds fast_align and GIZA++ into ./tools (needs cmake, a C++ compiler,
                                    # libgoogle-perftools-dev, libsparsehash-dev)
```

The embedding aligners (`embed`, `vote`, `grow`, `eval`) need only the pip packages. Set `COMBALIGN_TOOLS` if
fast_align and GIZA++ are built somewhere other than `./tools`.

## Usage

Input is tokenised text with one sentence per line. A line may start with an id followed by a tab, as in the gold
set. Alignments are written in Pharaoh format (`i-j`, 0-based).

```bash
# embedding aligners (no training data needed)
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
python -m combalign vote -k 3 -o vote.pharaoh labse8.pharaoh labse6.pharaoh nllb.pharaoh nllb8.pharaoh \
    eflomal.pharaoh fast_align.pharaoh giza.pharaoh

# grow: extend the vote with links from phrase-capable aligners (multiword alignments)
python -m combalign stat --aligner eflomal --sym gdfa ... -o eflomal.gdfa.pharaoh
python -m combalign grow --seed vote.pharaoh --cand-k 2 -o grow.pharaoh eflomal.gdfa.pharaoh fast_align.gdfa.pharaoh \
    nllb.itermax.pharaoh

# evaluate against a gold file (p = possible link, - = sure link)
python -m combalign eval --gold test_alignments.pharaoh vote.pharaoh grow.pharaoh
```

The `--model` option takes a name from `combalign/simalign.py` (`mbert`, `xlmr`, `xlmrL`, `labse`, `labse8`,
`labse12`, `nllb`, `nllb8`, `e5`), a Hugging Face model id, or a local directory such as a fine-tuned awesome-align
model. The extraction methods are `argmax`, `itermax`, `match` and `softmax` (awesome-align's).

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

## Reproducing the experiments

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
bash experiments/finetune_awesome.sh labse 40000                       # optional, GPU recommended
```

`experiments/extract_phrases.py` counts phrase pairs over a whole corpus (eflomal gdfa blocks with Dice scores),
for multiword-expression and bilingual dictionary induction.

## Citation

```bibtex
@inproceedings{steingrimsson-etal-2021-combalign,
  title     = {{CombAlign}: a Tool for Obtaining High-Quality Word Alignments},
  author    = {Steingr{\'i}msson, Stein{\th}{\'o}r and Loftsson, Hrafn and Way, Andy},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  year      = {2021}
}
```

Please also cite the aligners you use: fast_align (Dyer et al. 2013), eflomal (Östling & Tiedemann 2016), GIZA++
(Och & Ney 2003), SimAlign (Jalili Sabet et al. 2020), awesome-align (Dou & Neubig 2021), LaBSE (Feng et al. 2022)
and NLLB (NLLB Team 2022).
