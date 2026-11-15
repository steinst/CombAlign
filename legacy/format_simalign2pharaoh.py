import sys

infile = sys.argv[1]

with open(infile) as fi, open(infile + '.pharaoh', 'w') as fo:
    for line in fi:
        alignments = line.strip().split('), (')
        outline = ''
        for a in alignments:
            b = a.replace('[','').replace(']','').replace('(','').replace(')','').replace(', ', '-')
            outline += b + ' '
        fo.write(outline.strip() + '\n')
