# -*- coding: utf-8 -*-
"""Sentence pairs from ParIce 21.10 TMX (streamed from the zip) for the statistical aligners. Tokenised like the gold
set (punctuation split off), lowercased copies for training; dedup; length 1–80 tokens, ratio ≤ 3.
Output data/train/parice.{en,is} (all, shuffled with seed 1) — size subsets are prefixes (100k, 500k, 1M, all).
    python experiments/prep_parice.py <ParIce-train.21.10.zip from CLARIN-IS>"""
import os, random, re, sys, zipfile
import xml.etree.ElementTree as ET
ZIP = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/data/train", exist_ok=True)
OUT = f"{ROOT}/data/train/parice"
TOK = re.compile(r"\w+(?:[-'’.]\w+)*|[^\w\s]", re.U)
tok = lambda s: " ".join(TOK.findall(s))   # noqa: E731
seen, pairs = set(), []
with zipfile.ZipFile(ZIP) as z:
    for name in sorted(n for n in z.namelist() if n.endswith(".tmx")):
        k = 0
        with z.open(name) as f:
            for ev, el in ET.iterparse(f, events=("end",)):
                if el.tag != "tu":
                    continue
                seg = {}
                for tuv in el.findall("tuv"):
                    lang = (tuv.get("{http://www.w3.org/XML/1998/namespace}lang") or tuv.get("lang") or "")[:2].lower()
                    s = tuv.find("seg")
                    if s is not None:
                        seg[lang] = "".join(s.itertext()).strip()
                el.clear()
                if "en" in seg and "is" in seg:
                    e, i = tok(seg["en"]), tok(seg["is"])
                    ne, ni = len(e.split()), len(i.split())
                    if 1 <= ne <= 80 and 1 <= ni <= 80 and max(ne, ni) <= 3 * min(ne, ni) + 2 and (e, i) not in seen:
                        seen.add((e, i)); pairs.append((e, i)); k += 1
        print(name, k, flush=True)
random.Random(1).shuffle(pairs)
with open(OUT + ".en", "w") as fe, open(OUT + ".is", "w") as fi:
    for e, i in pairs:
        fe.write(e + "\n"); fi.write(i + "\n")
print("total", len(pairs))
