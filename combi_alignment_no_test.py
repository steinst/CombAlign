# -*- coding: UTF-8 -*-

import argparse
import collections
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('--combitype', '-m', help='type of alignment', choices=['unionall', 'unionallbut1', 'unionallbut2', 'unionallbut3', 'unionmin2', 'unionmin3'], default='unionall')
parser.add_argument('--output', '-o', default='combialign', help='output file name')
parser.add_argument('--alignment_folder', '-dir', required=True, help='alignment folder name')
parser.add_argument('--alignments', '-a', nargs='+', required=True, help='input file names')
parser.add_argument('--remove', '-r', default=0, help='combinations with alignments removed')

args = parser.parse_args()


def remove_one_from_list(data, index_to_remove):
    new_data = [v for i, v in enumerate(data) if i != index_to_remove]
    return new_data


config = configparser.ConfigParser()
config.read('config_notest.ini')

all_alignments = [args.alignments]

use_alignments = []
al_length = len(args.alignments)
temp = []
iterations = 0
while iterations < int(args.remove):
    for i in range(al_length):
        for j in all_alignments:
            use_alignments.append(remove_one_from_list(j, i))
    all_alignments = use_alignments.copy()
    use_alignments = []
    iterations += 1

check_combination = list(set(tuple(sorted(x)) for x in all_alignments))

for alignment_files in check_combination:
    print(alignment_files)
    num_files = len(alignment_files)
    manFile = open(args.alignment_folder + '/' + alignment_files[0], 'r')
    manLines = manFile.readlines()
    num_lines = len(manLines)
    manFile.close()

    inarray = [''] * num_lines
    outarray = [inarray] * num_files

    af_ctr = 0
    for i in alignment_files:
        manFile = open(args.alignment_folder + '/' + i, 'r')
        manLines = manFile.readlines()
        num_lines = len(manLines)
        manFile.close()

        for j in range(num_lines):
            current = manLines[j].strip()
            inarray[j] += current + ' '

    unionarray = [[]] * num_lines

    outfilename = args.alignment_folder + '/' + args.output + '_' + args.combitype + '.out'

    with open(outfilename, 'w') as out:
        alignments_out = ''
        for j in range(num_lines):
            if args.combitype == 'unionall':
                alignments_out = ' '.join(list(set(inarray[j].split())))
            elif args.combitype == 'unionallbut1':
                countedalignments = collections.Counter(inarray[j].split())
                unionbut1alignments = []
                for key, value in dict(countedalignments).items():
                    if value >= num_files - 1:
                        unionbut1alignments.append(key)
                        alignments_out = ' '.join(unionbut1alignments)
            elif args.combitype == 'unionallbut2':
                countedalignments = collections.Counter(inarray[j].split())
                unionbut2alignments = []
                for key, value in dict(countedalignments).items():
                    if value >= num_files - 2:
                        unionbut2alignments.append(key)
                        alignments_out = ' '.join(unionbut2alignments)
            elif args.combitype == 'unionallbut3':
                countedalignments = collections.Counter(inarray[j].split())
                unionbut2alignments = []
                for key, value in dict(countedalignments).items():
                    if value >= num_files - 3:
                        unionbut2alignments.append(key)
                        alignments_out = ' '.join(unionbut2alignments)
            elif args.combitype == 'unionmin2':
                countedalignments = collections.Counter(inarray[j].split())
                unionbut2alignments = []
                for key, value in dict(countedalignments).items():
                    if value >= 2:
                        unionbut2alignments.append(key)
                        alignments_out = ' '.join(unionbut2alignments)
            elif args.combitype == 'unionmin3':
                countedalignments = collections.Counter(inarray[j].split())
                unionbut2alignments = []
                for key, value in dict(countedalignments).items():
                    if value >= 3:
                        unionbut2alignments.append(key)
                        alignments_out = ' '.join(unionbut2alignments)
            out.write(alignments_out + '\n')
            alignments_out = ''
