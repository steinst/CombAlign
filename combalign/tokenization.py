# -*- coding: utf-8 -*-
"""Tokenisation with character offsets (`--tokenize`), and the output formats that use them.

  tokenize(text, lang)   → [(start, end)] character spans into `text`; a token is always text[start:end], so links can be
                           mapped back to the raw sentence. English: Moses (sacremoses); Icelandic: Miðeind `tokenizer`
                           (abbreviations, ordinals, dates); other languages: Moses with that language's rules.
  whitespace_spans(text) → spans of an already tokenised line (the default without --tokenize)
  project(...)           → move links from one tokenisation of a text to another through the character spans
  to_offsets / to_jsonl  → `--output-format offsets | jsonl`

The conventions follow the English–Icelandic gold set where it is consistent (see results/RESULTS.md, tokenisation):
  * English clitics are split as in the Penn Treebank: I 'll, it 's, do n't, ca n't (Moses alone gives don 't), also with
    a typographic apostrophe (he ’s);
  * a token never ends inside a run of letters and digits (CYP3A4, m2), and "/", ":" and "·" do not split between two
    digits (2626/84, 5:00, 0·005; but og / eða, mg / kg, 96/23 / EC);
  * multiword tokens of the Icelandic tokenizer (dates, amounts: "3. janúar 2020", "1.500 kr.") are split at the spaces,
    since the aligners work on words.
"""
import functools
import json
import re
import sys

_CLITICS = {"s", "ll", "re", "ve", "d", "m", "t"}
_APOS = "'’"


def whitespace_spans(text):
    return [m.span() for m in re.finditer(r"\S+", text)]


def spans_of(text, tokens):
    """Spans of `tokens` in `text`, when the tokens are the text with only whitespace added or removed.
    → [(start, end)] or None if they are not (the tokenizer changed characters)."""
    spans, pos = [], 0
    for tok in tokens:
        while pos < len(text) and (text[pos].isspace() or (ord(text[pos]) < 32)):
            pos += 1
        if not tok or not text.startswith(tok, pos):
            return None
        spans.append((pos, pos + len(tok)))
        pos += len(tok)
    return spans if not text[pos:].strip() else None


@functools.lru_cache(maxsize=None)
def _moses(lang):
    from sacremoses import MosesTokenizer
    return MosesTokenizer(lang=lang)


def _moses_spans(text, lang):
    toks = _moses(lang).tokenize(text, escape=False)
    return spans_of(text, toks)


def _mideind_spans(text):
    try:
        import tokenizer as mideind
    except ImportError:
        raise SystemExit("--tokenize for Icelandic uses the Miðeind tokenizer: pip install tokenizer") from None
    spans, pos = [], 0
    for t in mideind.tokenize(text):
        orig = t.original or ""
        if t.txt and orig.strip():
            lead = len(orig) - len(orig.lstrip())
            # a multiword token (date, amount, ...) is split at its spaces
            spans.extend((pos + lead + a, pos + lead + b) for a, b in whitespace_spans(orig.strip()))
        pos += len(orig)
    if pos > len(text) or text[pos:].strip() or "".join(text[a:b] for a, b in spans) != "".join(text.split()):
        return None                                  # the originals do not add up to the input: do not trust the offsets
    return spans


def _merge(text, spans, lang):
    """The shared conventions (module docstring), applied to adjacent spans."""
    out = []
    k = 0
    while k < len(spans):
        a, b = spans[k]
        tok = text[a:b]
        prev = out[-1] if out else None
        glued = prev is not None and prev[1] == a
        nxt = spans[k + 1] if k + 1 < len(spans) else None
        nxt_glued = nxt is not None and nxt[0] == b
        if lang == "en" and tok in _APOS and tok != "'" and nxt_glued and text[nxt[0]:nxt[1]].lower() in _CLITICS:
            a, b = a, nxt[1]                           # ’ + s → ’s  (Moses only knows the ASCII apostrophe)
            tok = text[a:b]
            k += 1
            nxt = spans[k + 1] if k + 1 < len(spans) else None
            nxt_glued = nxt is not None and nxt[0] == b
        if lang == "en" and len(tok) == 2 and tok[0] in _APOS and tok[1] in "tT" and glued and text[a - 1] in "nN":
            if prev[1] - prev[0] == 1:                 # n + 't → n't
                out.pop()
            else:                                      # don + 't → do + n't
                out[-1] = (prev[0], prev[1] - 1)
            out.append((a - 1, b))
        elif glued and text[a - 1].isalnum() and text[a].isalnum():
            out[-1] = (prev[0], b)                     # never split inside a run of letters and digits
        elif len(tok) == 1 and tok in "/:·\u0387" and glued and nxt_glued and text[a - 1].isdigit() and text[b].isdigit():
            out[-1] = (prev[0], nxt[1])                # 96/23 , 5:00
            k += 1
        else:
            out.append((a, b))
        k += 1
    return out


def tokenize(text, lang="en"):
    """→ [(start, end)] spans into `text`. Falls back to the whitespace tokens of the line (with a warning) if the
    tokenizer's output cannot be located in the text."""
    spans = _mideind_spans(text) if lang == "is" else _moses_spans(text, lang)
    if spans is None:
        print(f"[tokenize] could not map the {lang} tokens back to the text, using whitespace tokens: {text[:60]!r}",
              file=sys.stderr)
        return whitespace_spans(text)
    return _merge(text, spans, lang)


def tokens(text, spans):
    return [text[a:b] for a, b in spans]


def project(links, src_from, trg_from, src_to, trg_to):
    """Links between tokens with spans src_from / trg_from → links between the tokens with spans src_to / trg_to over
    the same texts: a link is carried to every pair of tokens its two ends overlap."""
    def overlaps(frm, to):
        return [[k for k, (a, b) in enumerate(to) if a < y and x < b] for x, y in frm]
    so, to = overlaps(src_from, src_to), overlaps(trg_from, trg_to)
    return {(i2, j2) for i, j in links for i2 in so[i] for j2 in to[j]}


def to_offsets(links, src_spans, trg_spans):
    """`--output-format offsets`: one line per sentence pair, a link is "s:e-s:e" = source characters [s, e) ↔ target
    characters [s, e) of the input lines (after the id and tab, if the line has one)."""
    return " ".join(f"{src_spans[i][0]}:{src_spans[i][1]}-{trg_spans[j][0]}:{trg_spans[j][1]}" for i, j in sorted(links))


def to_jsonl(links, src, trg, src_spans, trg_spans, sid=None, votes=None, n=None, grown=None, grown_counts=None, m=None):
    """`--output-format jsonl`: one object per sentence pair with the raw texts, tokens, character spans and links.
    A link is {"src": i, "trg": j} (token indices), plus "votes" and "of" when the vote count is known (c of n aligners)
    and "grown": true for links added by --grow (their "votes"/"of" then count the candidate aligners)."""
    out = []
    for i, j in sorted(links):
        d = {"src": i, "trg": j}
        if grown and (i, j) in grown:
            if grown_counts and (i, j) in grown_counts:
                d.update(votes=grown_counts[(i, j)], of=m)
            d["grown"] = True
        elif votes and votes.get((i, j)) is not None:
            d.update(votes=votes[(i, j)], of=n)
        out.append(d)
    rec = {} if sid is None else {"id": sid}
    rec.update(src=src, trg=trg, src_tokens=tokens(src, src_spans), trg_tokens=tokens(trg, trg_spans),
               src_spans=[list(s) for s in src_spans], trg_spans=[list(s) for s in trg_spans], links=out)
    return json.dumps(rec, ensure_ascii=False)
