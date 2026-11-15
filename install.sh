#!/bin/bash
# Builds the statistical aligners into ./tools and installs the Python dependencies into the active environment.
#   fast_align   https://github.com/clab/fast_align   (needs cmake, libgoogle-perftools-dev, libsparsehash-dev)
#   GIZA++       https://github.com/moses-smt/giza-pp
#   eflomal, transformers, torch, sacremoses, tokenizer (Miðeind, for --tokenize): pip (requirements.txt)
set -e
cd "$(dirname "$0")"
pip install -r requirements.txt
mkdir -p tools && cd tools
[ -x fast_align/build/fast_align ] || { rm -rf fast_align; git clone -q https://github.com/clab/fast_align.git && \
  mkdir -p fast_align/build && (cd fast_align/build && cmake .. > /dev/null && make -j8 > /dev/null); }
[ -x giza-pp/GIZA++-v2/GIZA++ ] || { rm -rf giza-pp; git clone -q https://github.com/moses-smt/giza-pp.git && \
  (cd giza-pp && make -j8 > build.log 2>&1); }
ls -la fast_align/build/fast_align giza-pp/GIZA++-v2/GIZA++ giza-pp/GIZA++-v2/snt2cooc.out giza-pp/mkcls-v2/mkcls
python -c "import eflomal; print('eflomal ok')"
