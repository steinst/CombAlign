# -*- coding: utf-8 -*-
"""CombAlign recipes as data: which aligners vote, how many votes are needed, and how `--grow` extends the vote.
Numbers (English–Icelandic gold set, test; configurations chosen on dev, see results/RESULTS.md):
  emb   LaBSE L8, LaBSE L6 κ.09, NLLB L12                                    vote 2/3  F1 .947 / AER .054   needs nothing
  fast  LaBSE L6 κ.09, NLLB L12, eflomal ∩                                   vote 2/3  F1 .954 / AER .046   needs a parallel corpus
  full  the 7 below                                                           vote 3/7  F1 .960 / AER .043   + GIZA++ (slow)
Members are the names used throughout: an encoder member is (model, layer, extraction method, level, κ), a statistical
member is (aligner, symmetrisation, training pairs used). Encoder members of one model share forward passes."""

MEMBERS = {
    # encoders (no training): the model key is a combalign.simalign.MODELS entry; layers as in the paper
    "labse8":        dict(kind="embed", model="labse", layer=8, method="argmax", level="word", kappa=0.0),
    "labse6k09":     dict(kind="embed", model="labse", layer=6, method="argmax", level="word", kappa=0.09),
    "nllb12":        dict(kind="embed", model="nllb", layer=12, method="argmax", level="word", kappa=0.0),
    "nllb8k09":      dict(kind="embed", model="nllb", layer=8, method="argmax", level="word", kappa=0.09),
    # phrase-capable encoder members (only used by --grow)
    "labse_itermax": dict(kind="embed", model="labse", layer=6, method="itermax", level="word", kappa=0.09),
    "nllb_itermax":  dict(kind="embed", model="nllb", layer=12, method="itermax", level="word", kappa=0.09),
    "nllb_softmax":  dict(kind="embed", model="nllb", layer=12, method="softmax", level="bpe", kappa=0.0),
    # statistical aligners, trained on the user's corpus + the sentences to align; GIZA++ on at most 1 M pairs
    "eflomal":         dict(kind="stat", aligner="eflomal", sym="intersect", n_train=None),
    "fast_align":      dict(kind="stat", aligner="fast_align", sym="intersect", n_train=None),
    "giza":            dict(kind="stat", aligner="giza", sym="intersect", n_train=1_000_000),
    "eflomal_gdfa":    dict(kind="stat", aligner="eflomal", sym="gdfa", n_train=None),
    "fast_align_gdfa": dict(kind="stat", aligner="fast_align", sym="gdfa", n_train=None),
}

# members: the voters; k: votes needed. grow: seed = vote ≥ seed_k over the members, candidates = links proposed by
# ≥ cand_k of the grow members, final-and off. full's grow is the paper experiment's; emb's and fast's were chosen on dev.
RECIPES = {
    "emb": dict(members=["labse8", "labse6k09", "nllb12"], k=2, needs_corpus=False,
                grow=dict(seed_k=2, cand_k=3, members=["labse_itermax", "nllb_itermax", "nllb_softmax"])),
    "fast": dict(members=["labse6k09", "nllb12", "eflomal"], k=2, needs_corpus=True,
                 grow=dict(seed_k=2, cand_k=4, members=["eflomal_gdfa", "labse_itermax", "nllb_itermax", "nllb_softmax"])),
    "full": dict(members=["labse8", "labse6k09", "nllb12", "nllb8k09", "eflomal", "fast_align", "giza"], k=3, needs_corpus=True,
                 grow=dict(seed_k=4, cand_k=3, members=["eflomal_gdfa", "fast_align_gdfa", "nllb_itermax", "labse_itermax",
                                                        "nllb_softmax"])),
}
