# -*- coding: utf-8 -*-
"""Encoder throughput (sentences per second, source + target side counted separately) for batched multi-layer
encoding. Needs a corpus with one tokenised sentence per line (e.g. data/train/parice.en).
    python experiments/bench_encode.py MODEL LAYERS CORPUS [N=2000] [DEVICE=cpu] [BATCH=64] [FP32=0]
    e.g. python experiments/bench_encode.py labse 6,8 data/train/parice.en 5000 cuda:0 64"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from combalign import simalign   # noqa: E402

if __name__ == "__main__":
    model, layers, corpus = sys.argv[1], tuple(int(x) for x in sys.argv[2].split(",")), sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    dev = sys.argv[5] if len(sys.argv) > 5 else "cpu"
    bs = int(sys.argv[6]) if len(sys.argv) > 6 else 64
    fp32 = len(sys.argv) > 7 and sys.argv[7] == "1"
    sents = []
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sents.append(line.split())
            if len(sents) >= n:
                break
    E = simalign.Embedder(model, device=dev, fp16=False if fp32 else None)
    E.encode(sents[:64], "en", layers, bs)                   # warm-up
    if dev.startswith("cuda"):
        E.torch.cuda.synchronize()
    t = time.time()
    E.encode(sents, "en", layers, bs)
    if dev.startswith("cuda"):
        E.torch.cuda.synchronize()
    dt = time.time() - t
    print(f"{model} layers {layers} {dev} batch {bs} {'fp32' if not E.fp16 else 'fp16'}: {len(sents)} sentences in {dt:.1f}s "
          f"= {len(sents) / dt:.0f} sent/s ({len(layers)} layers from one pass)")
