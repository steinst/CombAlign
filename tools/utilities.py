# -*- coding: UTF-8 -*-

import sys
import shutil
import subprocess
import os
from pathlib import Path

GOOGLE_SHEETS_CREDENTIAL_FILE = './client_secret.json'
secret_file = Path(GOOGLE_SHEETS_CREDENTIAL_FILE)


def send_data_to_google_sheet(sheet, row):
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    secret_file = Path(GOOGLE_SHEETS_CREDENTIAL_FILE)
    if secret_file.is_file():
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
        client = gspread.authorize(creds)

        # Open a Google Sheet, by name
        sheet = client.open("Word Alignment Results").worksheet(sheet)
        sheet.insert_row(row, 2)


def txt2fastalign(tok_folder, parallel_filename, fa_folder):
    print('Converting ' + parallel_filename + ' to fast-align format...')
    src_file = tok_folder + '/' + parallel_filename + '.src'
    trg_file = tok_folder + '/' + parallel_filename + '.trg'
    out_file = fa_folder + '/' + parallel_filename + '.fa'

    srcIn = open(src_file, 'r')
    srcLines = srcIn.readlines()
    srcIn.close()

    trgIn = open(trg_file, 'r')
    trgLines = trgIn.readlines()
    trgIn.close()

    with open(out_file, 'w') as fa_out:
        inLength = len(srcLines)
        print('Lines in file: ' + str(inLength))
        ctr = 0
        while ctr < inLength:
            lineOut = srcLines[ctr].strip() + ' ||| ' + trgLines[ctr].strip() + '\n'
            fa_out.write(lineOut)
            ctr += 1


def joinFiles(first_file, second_file):
    first_file_folder = first_file.rsplit('/', 1)[0]
    first_file_name = first_file.rsplit('/', 1)[1].rsplit('.', 1)[0]
    second_file_name = second_file.rsplit('/', 1)[1]
    outputfile = first_file_folder + '/' + first_file_name + '_' + second_file_name
    with open(outputfile, 'wb') as wfd:
        for f in [first_file, second_file]:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)
    return first_file_name + '_' + second_file_name


def align_fastalign(fastalign_folder, input_folder, sentence_alignments, output_folder_alignments, direction, testfile_out):
    if direction == 'fwd':
        bashCommand = fastalign_folder + '/fast_align -i ' + input_folder + '/' + sentence_alignments + " -d -o -v > " + output_folder_alignments + '/' + testfile_out + '.fwd'
    elif direction == 'rev':
        bashCommand = fastalign_folder + '/fast_align -i ' + input_folder + '/' + sentence_alignments + " -d -o -v -r > " + output_folder_alignments + '/' + testfile_out + '.rev'
    bashCommand = bashCommand.replace('//', '/')
    process = subprocess.run(bashCommand, shell=True)
    return output_folder_alignments + '/' + testfile_out + '.' + direction


def align_eflomal_makepriors(eflomal_folder, input_folder, training_corpus_name, output_folder_alignments):
    if os.path.exists(output_folder_alignments + '/eflomal_' + training_corpus_name + '.fwd'): os.remove(output_folder_alignments + '/eflomal_' + training_corpus_name + '.fwd')
    if os.path.exists(output_folder_alignments + '/eflomal_' + training_corpus_name + '.rev'): os.remove(output_folder_alignments + '/eflomal_' + training_corpus_name + '.rev')
    alignCommand = eflomal_folder + '/align.py --model 3 -i ' + input_folder + '/' + training_corpus_name + ".fa -f " + output_folder_alignments + '/eflomal_' + training_corpus_name + '.fwd -r ' + output_folder_alignments + '/eflomal_' + training_corpus_name + '.rev'
    alignCommand = alignCommand.replace('//', '/')
    process = subprocess.run(alignCommand, shell=True)
    priorCommandFwdRev = eflomal_folder + '/makepriors.py -i ' + input_folder + '/' + training_corpus_name + ".fa -f " + output_folder_alignments + '/eflomal_' + training_corpus_name + '.fwd -r ' + output_folder_alignments + '/eflomal_' + training_corpus_name + '.rev --priors ' + output_folder_alignments + '/eflomal_fr_' + training_corpus_name + '.priors'
    priorCommandFwdRev = priorCommandFwdRev.replace('//', '/')
    process = subprocess.run(priorCommandFwdRev, shell=True)


def align_gizapp_p2s(gizapp_folder, fa_folder, corpus_file):
    p2sCommand = gizapp_folder + '/plain2snt.out ' + fa_folder + '/' + corpus_file + '.src ' + fa_folder + '/' + corpus_file + '.trg '
    p2sCommand = p2sCommand.replace('//', '/')
    process = subprocess.run(p2sCommand, shell=True)


def align_gizapp_mkcls(gizapp_mkcls, n_value, fa_folder, corpus_file, direction):
    mkclsCommand = gizapp_mkcls + '/mkcls -n' + str(n_value) + ' -p' + fa_folder + '/' + corpus_file + '.' + direction + ' -V' + fa_folder + '/' + corpus_file + '.' + direction + '.vcb.classes'
    mkclsCommand = mkclsCommand.replace('//', '/')
    process = subprocess.run(mkclsCommand, shell=True)


def align_gizapp_cooc(gizapp_utils, fa_folder, corpus_file):
    coocCommand = gizapp_utils + '/snt2cooc.out' + ' ' + fa_folder + '/' + corpus_file + '.src.vcb ' + fa_folder + '/' + corpus_file + '.trg.vcb '+ fa_folder + '/' + corpus_file + '.src_' + corpus_file + '.trg.snt > ' + fa_folder + '/' + corpus_file + '.cooc'
    coocCommand = coocCommand.replace('//', '/')
    process = subprocess.run(coocCommand, shell=True)


def align_gizapp(gizapp_utils, fa_folder, corpus_file, output_folder_alignments):
    gizappCommand = gizapp_utils + '/GIZA++ -S '+ fa_folder + '/' + corpus_file + '.src.vcb -T ' + fa_folder + '/' + corpus_file + '.trg.vcb -C ' + fa_folder + '/' + corpus_file + '.src_' + corpus_file + '.trg.snt -o ' + corpus_file + ' -outputpath ' + output_folder_alignments + ' -CoocurrenceFile ' + corpus_file + '.cooc'
    gizappCommand = gizappCommand.replace('//', '/')
    process = subprocess.run(gizappCommand, shell=True)


def align_mgiza_p2s(mgiza_folder, fa_folder, corpus_file):
    p2sCommand = mgiza_folder + '/plain2snt ' + fa_folder + '/' + corpus_file + '.src ' + fa_folder + '/' + corpus_file + '.trg '
    p2sCommand = p2sCommand.replace('//', '/')
    process = subprocess.run(p2sCommand, shell=True)


def align_mgiza_mkcls(mgiza_folder, n_value, fa_folder, corpus_file, direction):
    mkclsCommand = mgiza_folder + '/mkcls -n' + str(n_value) + ' -p' + fa_folder + '/' + corpus_file + '.' + direction + ' -V' + corpus_file + '.vcb.classes'
    mkclsCommand = mkclsCommand.replace('//', '/')
    process = subprocess.run(mkclsCommand, shell=True)


def align_mgiza_cooc(mgiza_folder, fa_folder, output_folder_alignments, corpus_file):
    coocCommand = mgiza_folder + '/snt2cooc' + ' ' + output_folder_alignments + '/' + corpus_file + '.cooc ' + fa_folder + '/' + corpus_file + '.src.vcb ' + fa_folder + '/' + corpus_file + '.trg.vcb '+ fa_folder + '/' + corpus_file + '.src_' + corpus_file + '.trg.snt'
    coocCommand = coocCommand.replace('//', '/')
    process = subprocess.run(coocCommand, shell=True)


def align_mgiza(mgiza_folder, fa_folder, corpus_file, output_folder_alignments, ncpus):
    gizappCommand = mgiza_folder + '/mgiza -s '+ fa_folder + '/' + corpus_file + '.src.vcb -t ' + fa_folder + '/' + corpus_file + '.trg.vcb -c ' + fa_folder + '/' + corpus_file + '.src_' + corpus_file + '.trg.snt -o ' + corpus_file + ' -outputpath ' + output_folder_alignments + ' -coocurrenceFile ' + corpus_file + '.cooc -ncpus ' + str(ncpus)
    gizappCommand = gizappCommand.replace('//', '/')
    process = subprocess.run(gizappCommand, shell=True)


def align_mgiza_concat_output(corpus_file, output_folder_alignments, number):
    outputfile = output_folder_alignments + '/' + corpus_file + '.A3.final'
    filelist = []
    for i in range(number):
        filelist.append(output_folder_alignments + '/' + corpus_file + '.A3.final.part' + f'{i:03}')
    with open(outputfile, 'wb') as a3:
        for f in filelist:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, a3)


def sym_type2short(sym_type):
    if sym_type == 'grow-diag-final-and':
        sym_short = 'gdfa'
    elif sym_type == 'grow-diag':
        sym_short = 'gd'
    elif sym_type == 'grow-diag-final':
        sym_short = 'gdf'
    else:
        sym_short = sym_type
    return sym_short


def align_eflomal_syms(fastalign_folder, eflomal_folder, fa_folder, training_corpus_name, output_folder_alignments, sym_type):
    # rather use atools in the fast-align folder
    #sym_types = ['grow-diag', 'grow-diag-final', 'grow-diag-final-and', 'intersect', 'union']
    sym_short = sym_type2short(sym_type)

    symGdfaCommand = fastalign_folder + '/atools -c ' + sym_type + ' -i ' + output_folder_alignments + '/eflomal_' + training_corpus_name + '.fwd -j ' + output_folder_alignments + '/eflomal_' + training_corpus_name + '.rev > ' + output_folder_alignments + '/eflomal_' + sym_short + '_' + training_corpus_name + '.sym'
    symGdfaCommand = symGdfaCommand.replace('//', '/')
    process = subprocess.run(symGdfaCommand, shell=True)
    priorCommandGdfa = eflomal_folder + '/makepriors.py -i ' + fa_folder + '/' + training_corpus_name + ".fa -f " + output_folder_alignments + '/eflomal_' + sym_short + '_' + training_corpus_name + '.sym -r ' + output_folder_alignments + '/eflomal_' + sym_short + '_' + training_corpus_name + '.sym --priors ' + output_folder_alignments + '/eflomal_' + sym_short + '_' + training_corpus_name + '.priors'
    priorCommandGdfa = priorCommandGdfa.replace('//', '/')
    process = subprocess.run(priorCommandGdfa, shell=True)


def align_eflomal_tests(fastalign_folder, eflomal_folder, fa_folder, prior_sym_types, output_folder_alignments, training_corpus_name, testset_name):
    testfile_out = output_folder_alignments + '/eflomal__' + testset_name + '_' + training_corpus_name + '_____' + sym_type2short(prior_sym_types)
    try:
        if os.path.exists(testfile_out + '.fwd'): os.remove(testfile_out + '.fwd')
        if os.path.exists(testfile_out + '.rev'): os.remove(testfile_out + '.rev')
        testCommand = 'python3 ' + eflomal_folder + '/align.py -i ' + fa_folder + '/' + testset_name + '.fa --priors ' + output_folder_alignments + '/eflomal_' + sym_type2short(prior_sym_types) + '_' + training_corpus_name + '.priors --model 3 -f ' + testfile_out + '.fwd -r ' + testfile_out + '.rev'
        testCommand = testCommand.replace('//', '/')
        process = subprocess.run(testCommand, shell=True)
    except Exception as e:
        print(e)

    #sym_types = ['grow-diag', 'grow-diag-final', 'grow-diag-final-and', 'intersect', 'union']
    sym_types = ['intersect']

    for sym_type in sym_types:
        sym_short = sym_type2short(sym_type)

        symCommand = fastalign_folder + '/atools -c ' + sym_type + ' -i ' + testfile_out + '.fwd -j ' + testfile_out + '.rev > ' + testfile_out + '.' + sym_short
        symCommand = symCommand.replace('//', '/')
        process = subprocess.run(symCommand, shell=True)


def decouple_testset(outputFolder, inFile, numLines):
    try:
        filename = inFile.split(outputFolder)[1]
    except Exception as e:
        print(e)
        print(inFile)
        print(outputFolder)
        sys.exit(0)
    os.rename(inFile, inFile + '_complete')
    bashCommand = 'tail -n ' + str(numLines) + ' ' + inFile + '_complete > ' + inFile
    process = subprocess.run(bashCommand, shell=True)


def symmetrize_alignments(fastalign_folder, alignment_folder, set_name, fwd_file, rev_file, sym_type, model=''):
    print(fwd_file)
    outfile = fwd_file.rsplit('.', 1)[0]
    print(outfile)
    bashCommand = fastalign_folder + '/atools -i ' + fwd_file + ' -j ' + rev_file + ' -c ' + sym_type + ' > ' + outfile + '.' + sym_type
    process = subprocess.run(bashCommand, shell=True)
    return outfile + '.' + sym_type


def fasttext_matrix(inline, vecalign_model, direction):
    srcText = inline.split('|||')[0].strip().lstrip().split()
    trgText = inline.split('|||')[1].strip().lstrip().split()

    outfile = fwd_file.rsplit('.',1)[0]
    bashCommand = fastalign_folder + '/atools -i ' + fwd_file + ' -j ' + rev_file + ' -c ' + sym_type + ' > ' + outfile + '_' + model + '.' + sym_type
    process = subprocess.run(bashCommand, shell=True)
    return outfile + '.' + sym_type


def format_giza2pharaoh(output_folder, corpus_name, testlines, testfile_out):
    inFile = open(output_folder + '/' + corpus_name + '.A3.final', 'r')
    inLines = inFile.readlines()
    inFile.close()

    ctr = 1
    pharaohOut = ''
    pharaohDict = {}

    line_number = ''

    for line in inLines:
        if (ctr % 3) == 1:
            line_number = line.split('(')[1].split(')')[0]
        if (ctr % 3) == 0:
            pharaohLine = ''
            current = line.strip().split('})')
            tokenCtr = 0
            for token in current:
                try:
                    skipt = token.split('({')
                    if len(skipt) == 2:
                        try:
                            target_allt = skipt[1].strip().lstrip().split()
                        except:
                            target_allt = skipt[1].strip().lstrip()
                        for targ in target_allt:
                            if tokenCtr > 0:
                                pharaohLine += str(tokenCtr - 1) + '-' + str(int(targ) - 1) + ' '
                except Exception as e:
                    print('ERROR  ', e)
                    print(line)
                    print(token)
                tokenCtr += 1
            pharaohDict[line_number] = pharaohLine
        ctr += 1

    ctr = 0
    while ctr < int(line_number):
        ctr += 1
        if ctr <= (testlines):
            pharaohOut += pharaohDict[str(ctr)] + '\n'

    outFile = open(testfile_out + '.pharaoh', 'w')
    outFile.write(pharaohOut)
    outFile.close()

    return testfile_out + '.pharaoh'
   

def format_giza2pharaoh_notest(output_folder, corpus_name):
    inFile = open(output_folder + '/' + corpus_name + '.A3.final', 'r')
    inLines = inFile.readlines()
    inFile.close()

    ctr = 1
    pharaohOut = ''
    pharaohDict = {}

    line_number = ''

    for line in inLines:
        if (ctr % 3) == 1:
            line_number = line.split('(')[1].split(')')[0]
        if (ctr % 3) == 0:
            pharaohLine = ''
            current = line.strip().split('})')
            tokenCtr = 0
            for token in current:
                try:
                    skipt = token.split('({')
                    if len(skipt) == 2:
                        try:
                            target_allt = skipt[1].strip().lstrip().split()
                        except:
                            target_allt = skipt[1].strip().lstrip()
                        for targ in target_allt:
                            if tokenCtr > 0:
                                pharaohLine += str(tokenCtr - 1) + '-' + str(int(targ) - 1) + ' '
                except Exception as e:
                    print('ERROR  ', e)
                    print(line)
                    print(token)
                tokenCtr += 1
            pharaohDict[line_number] = pharaohLine
        ctr += 1

    with open(output_folder + '/' + corpus_name + '.pharaoh', 'w') as fo:
        ctr = 0
        while ctr < int(line_number):
            ctr += 1
            fo.write(pharaohDict[str(ctr)] + '\n')

def format_fa2giza(fa_folder, filename):
    inFile = open(fa_folder + '/' + filename + '.fa', 'r')
    inLines = inFile.readlines()
    inFile.close()

    outFile_src = fa_folder + '/' + filename + '.src'
    outFile_trg = fa_folder + '/' + filename + '.trg'

    with open(outFile_src, 'w') as out_s, open(outFile_trg, 'w') as out_t:
        for line in inLines:
            try:
                currLine = line.split('|||')
                srcLine = currLine[0].rstrip().strip() + '\n'
                trgLine = currLine[1].rstrip().strip() + '\n'
                out_s.write(srcLine)
                out_t.write(trgLine)
            except:
                pass
