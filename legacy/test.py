# -*- coding: UTF-8 -*-

import configparser
import argparse

parser = argparse.ArgumentParser()
# vantar awesome align
parser.add_argument('--method', '-m', help='test corpus', default='eflomal', choices=['eflomal', 'fastalign', 'simalign', 'gizapp', 'fasttext'])
parser.add_argument('--create-fa-format', '-fa', help='create fa-format files from training and testing files', action="store_true")
parser.add_argument('--generate-priors', '-gp', help='Generate priors (train eflomal)', action="store_true")
parser.add_argument('--training-corpus', '-tr', help='training corpus')
parser.add_argument('--align-file', '-align', help='sentences to align')
parser.add_argument('--src-lines', '-src', help='')
parser.add_argument('--trg-lines', '-trg', help='')
parser.add_argument("--matching-methods", type=str, default="a", help="m: Max Weight Matching (mwmf), a: argmax (inter), i: itermax, f: forward (fwd), r: reverse (rev)")
parser.add_argument("--token-type", type=str, choices=["bpe", "word"], default="word")
parser.add_argument("--num-align-sents", type=int, default=None, help="None means all sentences")
parser.add_argument("--distortion", type=float, default=0.09)
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--null-align", type=float, default=1.0)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("-log", action="store_true")
parser.add_argument('--model', type=str, default="bert", help="choices: ['bert', 'xlmr', 'xlml', '<transformer_model_name>']")
parser.add_argument("--output", type=str, default="simalign", help="output alignment files (without extension)")
# GIZA++
parser.add_argument('--input-fa', '-ifa', help='read in fa format', action="store_true")
parser.add_argument('--create-snt-format', '-p2s', help='create correct sentence format for giza', default=True)
parser.add_argument('--n-value', '-n', help='n-value for mkcls')
parser.add_argument('--use_training_corpus', default=True)

info = vars(parser.parse_args())

config = configparser.ConfigParser()
config.read('/home/steinst/PycharmProjects/CombAlign/config.ini')

for section in config.sections():
    info[section] = {}
    info[section].update(dict(config._sections[section]))
