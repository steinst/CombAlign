# -*- coding: utf-8 -*-
"""SimAlign-style alignment from contextual embeddings (Jalili Sabet et al. 2020), re-implemented for CombAlign
(no dependency on the simalign package). Models: mBERT (bert-base-multilingual-cased) and XLM-R
(xlm-roberta-base), layer 8 as in SimAlign; "word" = mean of a word's subword vectors, "bpe" = subword level mapped
back to words (any aligned subword pair aligns the words).
Methods: ArgMax (mutual argmax), IterMax (ArgMax repeated on the not-yet-aligned rows/columns as in
simalign iter_max, max_count 2, α 0.9), Match (maximum-weight bipartite matching, scipy). Distortion correction: similarity minus
κ · (i/m − j/n)² before extraction (κ = 0 off; the paper's best was 0.09).
"""
import re

import numpy as np

# name → (HF id, layer). Layers: SimAlign 8 for base models; LaBSE 6 (AccAlign's best vanilla layer); NLLB encoder
# 12 = last (TransAlign's best); XLM-R large 16 (≈ 2/3 depth, as 8/12 for base).
MODELS = {
    "mbert": ("bert-base-multilingual-cased", 8),
    "xlmr": ("xlm-roberta-base", 8),
    "xlmrL": ("xlm-roberta-large", 16),
    "labse": ("setu4993/LaBSE", 6), "labse8": ("setu4993/LaBSE", 8), "labse12": ("setu4993/LaBSE", 12),
    "nllb": ("facebook/nllb-200-distilled-600M", 12), "nllb8": ("facebook/nllb-200-distilled-600M", 8),
    "e5": ("intfloat/multilingual-e5-base", 8),
}
# ISO 639-1 → NLLB-200 (FLORES-200) language code; any other language: pass the NLLB code itself (e.g. --trg-lang fao_Latn)
NLLB_LANG = {
    "en": "eng_Latn", "is": "isl_Latn", "da": "dan_Latn", "sv": "swe_Latn", "nb": "nob_Latn", "no": "nob_Latn",
    "nn": "nno_Latn", "fo": "fao_Latn", "fi": "fin_Latn", "et": "est_Latn", "de": "deu_Latn", "nl": "nld_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "pt": "por_Latn", "it": "ita_Latn", "ca": "cat_Latn", "ro": "ron_Latn",
    "pl": "pol_Latn", "cs": "ces_Latn", "sk": "slk_Latn", "sl": "slv_Latn", "hr": "hrv_Latn", "hu": "hun_Latn",
    "lt": "lit_Latn", "lv": "lvs_Latn", "ga": "gle_Latn", "cy": "cym_Latn", "el": "ell_Grek", "bg": "bul_Cyrl",
    "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "tr": "tur_Latn",
}


def nllb_code(lang):
    if lang in NLLB_LANG:
        return NLLB_LANG[lang]
    if re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", lang):
        return lang
    raise SystemExit(f"no NLLB language code known for '{lang}': give the NLLB-200 code itself, e.g. deu_Latn "
                     f"(known short codes: {' '.join(sorted(NLLB_LANG))})")


class Embedder:
    def __init__(self, model="xlmr", device=None, layer=None, fp16=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        # a MODELS key, or any Hugging Face id / local directory (layer default 8)
        hf, default_layer = MODELS.get(model, (model, 8))
        self.torch, self.layer = torch, layer or default_layer
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nllb = "nllb" in hf
        self.tok = AutoTokenizer.from_pretrained(hf)
        m = AutoModel.from_pretrained(hf)
        self.m = (m.get_encoder() if self.nllb else m).to(self.dev).eval()
        self.fp16 = (self.dev.startswith("cuda") if fp16 is None else fp16) and self.dev.startswith("cuda")
        if self.fp16:                            # half precision on GPU only; vectors are averaged/normalised in fp32
            self.m = self.m.half()
        self.special = set(self.tok.all_special_ids)

    def __call__(self, words, lang="en"):
        """words: list of str → (word vectors [n×d] normalised, subword vectors normalised, subword→word index,
        raw subword vectors (for awesome-align's softmax extraction))."""
        if self.nllb:
            self.tok.src_lang = nllb_code(lang)
        enc = self.tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            h = self.m(input_ids=enc["input_ids"].to(self.dev), attention_mask=enc["attention_mask"].to(self.dev),
                       output_hidden_states=True).hidden_states[self.layer][0]
        wid = enc.word_ids()
        keep = [t for t, w in enumerate(wid) if w is not None and (not self.nllb or enc["input_ids"][0, t] not in
                                                                   self.tok.all_special_ids)]
        sub = h[keep]
        owner = [wid[t] for t in keep]
        wv = []
        for k in range(len(words)):
            idx = [n for n, o in enumerate(owner) if o == k]
            wv.append(sub[idx].mean(0) if idx else h[0] * 0)
        nz = lambda x: self.torch.nn.functional.normalize(x, dim=-1).cpu().numpy()   # noqa: E731
        return nz(self.torch.stack(wv)), nz(sub), owner, sub.cpu().numpy()

    def _pack(self, h, keep, owner, n_words):
        """One sentence of one layer: h [tokens × d] → the same 4-tuple as __call__ (fp32)."""
        torch = self.torch
        sub = h[keep].float()
        if n_words == 0:
            z = torch.zeros(0, h.shape[-1])
            return z.numpy(), sub.cpu().numpy(), owner, sub.cpu().numpy()
        own = torch.tensor(owner, dtype=torch.long, device=sub.device)
        wv = torch.zeros(n_words, sub.shape[-1], device=sub.device).index_add_(0, own, sub)
        wv = wv / torch.bincount(own, minlength=n_words).clamp(min=1).unsqueeze(1)      # unowned words stay 0
        nz = lambda x: torch.nn.functional.normalize(x, dim=-1).cpu().numpy()   # noqa: E731
        return nz(wv), nz(sub), owner, sub.cpu().numpy()

    def encode(self, sentences, lang="en", layers=None, batch_size=64):
        """Batched encoding. sentences: list of word lists. One forward pass per batch returns every requested layer
        (e.g. LaBSE L6 and L8, NLLB L8 and L12 cost one pass). Sentences are sorted by length, padded with an
        attention mask, and returned in the input order. → {layer: [4-tuple as __call__ per sentence]}."""
        torch = self.torch
        layers = tuple(dict.fromkeys(layers)) if layers else (self.layer,)
        if self.nllb:
            self.tok.src_lang = nllb_code(lang)
        n = len(sentences)
        order = sorted(range(n), key=lambda i: sum(len(w) + 1 for w in sentences[i]))
        out = {L: [None] * n for L in layers}
        for b in range(0, n, batch_size):
            idx = order[b:b + batch_size]
            enc = self.tok([sentences[i] for i in idx], is_split_into_words=True, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
            with torch.no_grad():
                hs = self.m(input_ids=enc["input_ids"].to(self.dev), attention_mask=enc["attention_mask"].to(self.dev),
                            output_hidden_states=True).hidden_states
            for r, i in enumerate(idx):
                wid, ids = enc.word_ids(r), enc["input_ids"][r].tolist()
                keep = [t for t, w in enumerate(wid) if w is not None and (not self.nllb or ids[t] not in self.special)]
                owner = [wid[t] for t in keep]
                for L in layers:
                    out[L][i] = self._pack(hs[L][r], keep, owner, len(sentences[i]))
        return out


def _distort(sim, kappa):
    if not kappa:
        return sim
    m, n = sim.shape
    pi = (np.arange(m)[:, None] / max(1, m - 1)) - (np.arange(n)[None, :] / max(1, n - 1))
    return sim - kappa * pi ** 2


def argmax(sim):
    a, b = sim.argmax(1), sim.argmax(0)
    return {(i, int(a[i])) for i in range(sim.shape[0]) if b[a[i]] == i}


def itermax(sim, max_count=2, alpha=0.9):
    """As simalign.SentenceAligner.iter_max: extra rounds of mutual argmax on sim × α·(row unaligned + col unaligned);
    cells whose row and column are both aligned are excluded."""
    m_, n_ = sim.shape
    inter = np.zeros_like(sim)
    for i, j in argmax(sim):
        inter[i, j] = 1
    for _ in range(max_count - 1):
        mx = 1.0 - inter.sum(1, keepdims=True).clip(0, 1)
        my = 1.0 - inter.sum(0, keepdims=True).clip(0, 1)
        if mx.sum() < 1 or my.sum() < 1:
            break
        mask = alpha * mx + alpha * my
        zeros = 1.0 - (1.0 - mx) * (1.0 - my)
        new = sim * mask
        fwd = np.zeros_like(sim); fwd[np.arange(m_), new.argmax(1)] = 1
        bac = np.zeros_like(sim); bac[new.argmax(0), np.arange(n_)] = 1
        prev = inter.sum()
        inter = (inter + fwd * bac * zeros).clip(0, 1)
        if inter.sum() == prev:
            break
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(inter))}


def match(sim):
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-sim)
    return set(zip(r.tolist(), c.tolist()))


def softmax_thr(dot, thr=1e-3):
    """awesome-align (Dou & Neubig 2021) extraction: softmax of raw dot products over rows and over columns,
    keep (i, j) where both probabilities exceed thr."""
    def sm(x, ax):
        x = x - x.max(ax, keepdims=True)
        e = np.exp(x)
        return e / e.sum(ax, keepdims=True)
    keep = (sm(dot, 1) > thr) & (sm(dot, 0) > thr)
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(keep))}


METHODS = {"argmax": argmax, "itermax": itermax, "match": match}


def align_pair(emb_s, emb_t, method="argmax", level="word", kappa=0.0):
    ws, ss, so, rs = emb_s
    wt, st, to, rt = emb_t
    if not len(ws) or not len(wt) or not len(ss) or not len(st):      # an empty sentence has no links
        return set()
    if method == "softmax":                      # always subword level on raw vectors, as awesome-align
        sub = softmax_thr(rs @ rt.T)
        return {(so[i], to[j]) for i, j in sub}
    if level == "word":
        return METHODS[method](_distort(ws @ wt.T, kappa))
    sub = METHODS[method](_distort(ss @ st.T, kappa))
    return {(so[i], to[j]) for i, j in sub}


def align_corpus_multi(src, trg, specs, embedder, src_lang="en", trg_lang="is", batch_size=64, chunk=2000):
    """Several members of one model from shared encoder passes. specs: [(layer, method, level, kappa)] → one list of
    alignments per spec. Pairs are processed in chunks of `chunk` so memory stays flat on large corpora."""
    layers = tuple(dict.fromkeys(sp[0] for sp in specs))
    out = [[] for _ in specs]
    for c in range(0, len(src), chunk):
        es = embedder.encode(src[c:c + chunk], src_lang, layers, batch_size)
        et = embedder.encode(trg[c:c + chunk], trg_lang, layers, batch_size)
        for o, (layer, method, level, kappa) in zip(out, specs):
            o.extend(align_pair(a, b, method, level, kappa) for a, b in zip(es[layer], et[layer]))
    return out


def align_corpus(src, trg, model="xlmr", method="argmax", level="word", kappa=0.0, embedder=None, cache=None,
                 layer=None, src_lang="en", trg_lang="is", batch_size=64, chunk=2000):
    """src, trg: lists of token lists. cache: dict for embeddings (reuse across methods/κ; keeps every sentence in
    memory). Without a cache the corpus is processed in chunks of `chunk` pairs."""
    E = embedder or Embedder(model)
    layer = layer or E.layer
    if cache is None:
        return align_corpus_multi(src, trg, [(layer, method, level, kappa)], E, src_lang, trg_lang, batch_size, chunk)[0]
    miss = [k for k in range(len(src)) if (model, layer, k) not in cache]
    if miss:
        es = E.encode([src[k] for k in miss], src_lang, (layer,), batch_size)[layer]
        et = E.encode([trg[k] for k in miss], trg_lang, (layer,), batch_size)[layer]
        for k, a, b in zip(miss, es, et):
            cache[(model, layer, k)] = (a, b)
    return [align_pair(*cache[(model, layer, k)], method, level, kappa) for k in range(len(src))]
