import sys
from sacremoses import MosesTokenizer

lang = sys.argv[1]
filein = sys.argv[2]
mt = MosesTokenizer(lang=lang)

with open(filein) as fi, open(filein + '.tokenized', 'w') as fo:
    for line in fi:
        tokenized = mt.tokenize(line, return_str=True)
        fo.write(tokenized + '\n')