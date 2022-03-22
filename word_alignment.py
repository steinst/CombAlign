# -*- coding: UTF-8 -*-

import configparser
import argparse
import time
from itertools import cycle
from AlignmentTools import Simalign, eflomal, fastalign, mgiza, gizapp
from tools import utilities

spinner = cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')


# fa-format is assumed

parser = argparse.ArgumentParser()
# vantar awesome align
parser.add_argument('--method', '-m', help='test corpus', default='eflomal', choices=['eflomal', 'fastalign', 'simalign', 'gizapp', 'awesome', 'mgiza'])
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
parser.add_argument('--n-value', '-n', default=10, help='n-value for mkcls')
parser.add_argument('--use_training_corpus', action="store_true")

info = vars(parser.parse_args())

config = configparser.ConfigParser()
config.read('config.ini')

for section in config.sections():
    info[section] = {}
    info[section].update(dict(config._sections[section]))

info['start_time'] = time.time()

#== preprocess data

#if args.training_corpus:
#    training_corpus_name = args.training_corpus

#if args.align_file:
#    alignment_set_name = args.align_file

testNumLines = 0
trainNumLines = 0
#sym_types = ['grow-diag', 'grow-diag-final', 'grow-diag-final-and', 'intersect', 'union']
sym_types = ['intersect']

try:
    if len(info['training_corpus']) > 0:
        info['DATA']['trainingcorpus'] = info['training_corpus']
except:
    pass

if len(info['align_file']) > 0:
    info['DATA']['alignmentset'] = info['align_file']

if info['create_fa_format']:
    print('Creating .fa format files...')
    if info['use_training_corpus']:
        utilities.txt2fastalign(info['FILE_FOLDERS']['training'], info['DATA']['trainingcorpus'], info['FILE_FOLDERS']['fa'])
    utilities.txt2fastalign(info['FILE_FOLDERS']['tokenized'], info['DATA']['alignmentset'], info['FILE_FOLDERS']['fa'])


if info['use_training_corpus']:
    fa2align = utilities.joinFiles(info['FILE_FOLDERS']['fa'] + '/' + info['DATA']['trainingcorpus'] + '.fa', info['FILE_FOLDERS']['fa'] + '/' + info['align_file'] + '.fa')
    srcIn = open(info['FILE_FOLDERS']['fa'] + '/' + info['DATA']['alignmentset'] + '.fa', 'r')
    srcLines = srcIn.readlines()
    srcIn.close()
    testNumLines = len(srcLines)
    trainIn = open(info['FILE_FOLDERS']['fa'] + '/' + info['DATA']['trainingcorpus'] + '.fa', 'r')
    trainLines = trainIn.readlines()
    trainIn.close()
    trainNumLines = len(trainLines)
else:
    fa2align = info['DATA']['alignmentset'] + '.fa'

info['fa2align'] = fa2align
info['num_test_lines'] = testNumLines
info['num_ttrain_lines'] = trainNumLines

#############################################
########### Training
#############################################
alignment_files = []

def days_hours_minutes(td, multiple=1):
    td = td*multiple
    return str(td.seconds//3600 + td.days*24).zfill(2), str((td.seconds//60)%60).zfill(2), str(td.seconds%60).zfill(2)

if info['method'] == 'simalign':
    alignment_object = Simalign(info)
    alignment_object.align()

elif info['method'] == 'eflomal':
    alignment_object = eflomal(info)
    alignment_object.align()

elif info['method'] == 'fastalign':
    alignment_object = fastalign(info)
    alignment_object.align()

elif info['method'] == 'mgiza':
    alignment_object = mgiza(info)
    alignment_object.align()

elif info['method'] == 'gizapp':
    alignment_object = gizapp(info)
    alignment_object.align()


#total_time = str(time.time() - start_time)
#print(total_time)
