# -*- coding: utf-8 -*-
"""CombAlign command line. Sentence files: one tokenised sentence per line (optionally "<id>\\t<sentence>").
Alignments: Pharaoh format, one line per sentence pair (i-j = source word i ↔ target word j, 0-based).

  python -m combalign align --recipe emb|fast|full [--grow] --src x.en --trg x.is [--train-src c.en --train-trg c.is] \\
                            [--src-lang en --trg-lang is] [--device cuda] [--fp32] [--work DIR] [--scores s.txt] \\
                            [--min-votes N] -o out.pharaoh
  python -m combalign stat  --aligner eflomal --src x.en --trg x.is [--train-src c.en --train-trg c.is] \\
                            [--sym intersect|union|gd|gdf|gdfa] -o eflomal.pharaoh
  python -m combalign embed --model labse8 [--method argmax|itermax|match|softmax] [--level word|bpe] \\
                            [--kappa 0.0] [--device cuda] [--batch-size 64] [--fp32] --src x.en --trg x.is -o labse8.pharaoh
  python -m combalign vote  -k 3 [--scores vote.scores] [--min-votes 4] -o vote.pharaoh a.pharaoh b.pharaoh c.pharaoh ...
  python -m combalign grow  --seed vote.pharaoh [--seed-scores vote.scores] --cand-k 3 [--scores grow.scores] \\
                            -o grow.pharaoh p1.pharaoh p2.pharaoh ...
  python -m combalign filter --scores x.scores --min-votes 4 [--no-grown] -o x.min4.pharaoh
  python -m combalign eval  --gold gold.pharaoh pred1.pharaoh [pred2.pharaoh ...]
"""
import argparse
import sys

from . import core


def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="combalign", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    al = sp.add_parser("align", help="one command: run a recipe (emb | fast | full), optionally with grow")
    al.add_argument("--recipe", choices=("emb", "fast", "full"), required=True,
                    help="emb: 3 encoders, no training | fast: 2 encoders + eflomal | full: 4 encoders + eflomal, fast_align, GIZA++")
    al.add_argument("--grow", action="store_true", help="grow the vote into multiword (1-n, n-m) blocks")
    al.add_argument("--src", required=True); al.add_argument("--trg", required=True)
    al.add_argument("--train-src"); al.add_argument("--train-trg", help="parallel corpus for fast/full (not needed for emb)")
    al.add_argument("--src-lang", default="en"); al.add_argument("--trg-lang", default="is")
    al.add_argument("--device"); al.add_argument("--batch-size", type=int, default=64)
    al.add_argument("--fp32", action="store_true", help="no half precision on GPU (reproduces the paper numbers exactly)")
    al.add_argument("--work", help="working directory: member outputs are cached here, so an interrupted run resumes "
                                   "(default: a temporary one)")
    al.add_argument("--scores", help='confidence file: "i-j:c/n" per link (proposed by c of n members); grown links "i-j:g3/5"')
    al.add_argument("--min-votes", type=int, help="require at least this many votes (for --grow: of the seed)")
    al.add_argument("-o", "--output", required=True)

    s = sp.add_parser("stat", help="statistical aligner (fast_align, eflomal, giza), both directions + symmetrisation")
    s.add_argument("--aligner", choices=("fast_align", "eflomal", "giza"), required=True)
    s.add_argument("--src", required=True); s.add_argument("--trg", required=True)
    s.add_argument("--train-src"); s.add_argument("--train-trg")
    s.add_argument("--n-train", type=int, help="use only the first N training pairs")
    s.add_argument("--sym", default="intersect", choices=("intersect", "union", "gd", "gdf", "gdfa", "fwd", "rev"))
    s.add_argument("--work", help="working directory (default: a temporary one)")
    s.add_argument("-o", "--output", required=True)

    e = sp.add_parser("embed", help="SimAlign-style aligner from contextual embeddings")
    e.add_argument("--model", default="labse8", help="a name from combalign.simalign.MODELS, a HF id or a local dir")
    e.add_argument("--layer", type=int)
    e.add_argument("--method", default="argmax", choices=("argmax", "itermax", "match", "softmax"))
    e.add_argument("--level", default="word", choices=("word", "bpe"))
    e.add_argument("--kappa", type=float, default=0.0, help="distortion correction (0 = off)")
    e.add_argument("--src-lang", default="en"); e.add_argument("--trg-lang", default="is")
    e.add_argument("--device")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--fp32", action="store_true", help="no half precision on GPU (default: fp16 on GPU)")
    e.add_argument("--src", required=True); e.add_argument("--trg", required=True)
    e.add_argument("-o", "--output", required=True)

    v = sp.add_parser("vote", help="keep links proposed by at least k of the input alignments")
    v.add_argument("-k", type=int, required=True)
    v.add_argument("--scores", help='write a confidence file: per sentence "i-j:c/n" (link proposed by c of n inputs)')
    v.add_argument("--min-votes", type=int, help="drop links with fewer than this many votes (≥ k) from the output")
    v.add_argument("alignments", nargs="+")
    v.add_argument("-o", "--output", required=True)

    g = sp.add_parser("grow", help="grow a high-precision seed with neighbouring candidate links (1-n / n-m blocks)")
    g.add_argument("--seed", required=True, help="seed alignment, e.g. the output of `vote`")
    g.add_argument("--cand-k", type=int, default=2, help="candidate links = proposed by ≥ k of the given alignments")
    g.add_argument("--no-final-and", action="store_true")
    g.add_argument("--seed-scores", help="scores file of the seed (from `vote --scores`): gives seed links their vote counts")
    g.add_argument("--scores", help='confidence file: seed links "i-j:c/n", grown links "i-j:g3/5" (3 of 5 candidate inputs)')
    g.add_argument("--min-votes", type=int, help="drop seed links with fewer votes (needs --seed-scores); grown links stay")
    g.add_argument("alignments", nargs="+", help="phrase-capable aligners (e.g. eflomal/fast_align gdfa, IterMax)")
    g.add_argument("-o", "--output", required=True)

    f = sp.add_parser("filter", help="trade recall for precision without re-running: Pharaoh from a scores file")
    f.add_argument("--scores", required=True)
    f.add_argument("--min-votes", type=int, default=1, help="keep voted links with at least this many votes")
    f.add_argument("--no-grown", action="store_true", help="drop links added by grow")
    f.add_argument("-o", "--output", required=True)

    ev = sp.add_parser("eval", help="precision, recall, F1, AER against a gold Pharaoh file (p = possible link)")
    ev.add_argument("--gold", required=True)
    ev.add_argument("predictions", nargs="+")

    a = ap.parse_args(argv)
    if a.cmd == "align":
        from . import pipeline
        src, trg = core.read_lines(a.src), core.read_lines(a.trg)
        r = pipeline.run(a.recipe, src, trg, a.train_src, a.train_trg, a.src_lang, a.trg_lang, a.grow, a.device,
                         False if a.fp32 else None, a.work, a.batch_size, a.min_votes)
        core.write_pharaoh(a.output, r["alignments"])
        if a.scores:
            lines = []
            for k, al in enumerate(r["alignments"]):
                grown = r["grown"][k] if r["grown"] else set()
                voted = {e: r["counts"][k][e] for e in al - grown}
                lines.append(core.to_scores(voted, r["n"], grown, r["grown_counts"][k] if r["grown"] else None, r["m"]))
            _write_lines(a.scores, lines)
    elif a.cmd == "stat":
        from . import external
        src, trg = core.read_lines(a.src), core.read_lines(a.trg)
        fwd, rev = external.ALIGNERS[a.aligner](src, trg, a.train_src, a.train_trg, a.n_train, a.work)
        out = fwd if a.sym == "fwd" else rev if a.sym == "rev" else [core.symmetrize(f, r, a.sym) for f, r in zip(fwd, rev)]
        core.write_pharaoh(a.output, out)
    elif a.cmd == "embed":
        from . import simalign as sa
        src, trg = core.read_lines(a.src), core.read_lines(a.trg)
        E = sa.Embedder(a.model, device=a.device, layer=a.layer, fp16=False if a.fp32 else None)
        out = sa.align_corpus(src, trg, a.model, a.method, a.level, a.kappa, embedder=E, src_lang=a.src_lang,
                              trg_lang=a.trg_lang, batch_size=a.batch_size)
        core.write_pharaoh(a.output, out)
    elif a.cmd == "vote":
        counts = core.vote([core.read_pharaoh(p) for p in a.alignments], a.k, return_counts=True)
        n = len(a.alignments)
        if a.min_votes:
            counts = [{e: c for e, c in d.items() if c >= a.min_votes} for d in counts]
        core.write_pharaoh(a.output, [set(d) for d in counts])
        if a.scores:
            _write_lines(a.scores, [core.to_scores(d, n) for d in counts])
    elif a.cmd == "grow":
        seed = core.read_pharaoh(a.seed)
        n_cand = len(a.alignments)
        cand_counts = core.vote([core.read_pharaoh(p) for p in a.alignments], a.cand_k, return_counts=True)
        seed_counts = n_seed = None
        if a.seed_scores:
            parsed = [core.parse_scores(l) for l in open(a.seed_scores, encoding="utf-8")]
            if len(parsed) != len(seed):
                sys.exit("--seed-scores has a different number of lines than --seed")
            seed_counts = [{e: v[0] for e, v in d.items()} for d in parsed]
            n_seed = next((v[1] for d in parsed for v in d.values() if v[1]), None)
            if a.min_votes:
                seed = [{e for e in s if (seed_counts[k].get(e) or 0) >= a.min_votes} for k, s in enumerate(seed)]
        elif a.min_votes:
            sys.exit("--min-votes for grow needs --seed-scores")
        res = [core.grow(s, set(c), not a.no_final_and, return_added=True) for s, c in zip(seed, cand_counts)]
        core.write_pharaoh(a.output, [al for al, _ in res])
        if a.scores:
            lines = []
            for k, (al, added) in enumerate(res):
                votes = {e: (seed_counts[k].get(e) if seed_counts else None) for e in al - added}
                lines.append(core.to_scores(votes, n_seed, added, cand_counts[k], n_cand))
            _write_lines(a.scores, lines)
    elif a.cmd == "filter":
        parsed = [core.parse_scores(l) for l in open(a.scores, encoding="utf-8")]
        core.write_pharaoh(a.output, [core.filter_scores(d, a.min_votes, not a.no_grown) for d in parsed])
    elif a.cmd == "eval":
        gold = [core.parse_pharaoh(l) for l in open(a.gold, encoding="utf-8")]
        for p in a.predictions:
            r = core.evaluate(core.read_pharaoh(p), gold)
            print(f"{p}\tP {r['P']:.3f}\tR {r['R']:.3f}\tF1 {r['F1']:.3f}\tAER {r['AER']:.3f}\tlinks {r['edges']}")


if __name__ == "__main__":
    sys.exit(main())
