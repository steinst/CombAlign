import os.path

from simalign import SentenceAligner
import datetime
from itertools import cycle
from tools import utilities

spinner = cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')


def update_progress_notice(ctr, num_lines, start_time, estimatedtime):
    now_time = datetime.datetime.now()
    print(" ", next(spinner), "{}/{}".format(str(ctr), str(num_lines)),
          ("  {}/{}".format(str(now_time - start_time), estimatedtime) if ctr % 100 == 0 else ""), end='\r')


def days_hours_minutes(td, multiple=1):
    td = td*multiple
    return str(td.seconds//3600 + td.days*24).zfill(2), str((td.seconds//60)%60).zfill(2), str(td.seconds%60).zfill(2)


def get_line_count(fa_file):
    print('Counting lines')
    srcIn = open(fa_file, 'r')
    srcLines = srcIn.readlines()
    srcIn.close()
    return len(srcLines)


class Simalign:
    #laga hvernig unnið er með matching methods
    def __init__(self, info_dict):
        self.num_lines = sum(1 for _ in open(info_dict['src_lines']))
        self.src_file = info_dict['src_lines']
        self.trg_file = info_dict['trg_lines']
        self.outfile = os.path.basename(info_dict['src_lines']).replace('.src','')
        self.matching_methods = info_dict['matching_methods']
        self.output_folder_alignments = info_dict['FILE_FOLDERS']['output']
        self.ctr = 0
        self.model = SentenceAligner(model=info_dict['model'], token_type=info_dict['token_type'], matching_methods=self.matching_methods, device=info_dict['device'])

    def align(self):
        with open(self.src_file) as src, open(self.trg_file) as trg, open(self.output_folder_alignments.rstrip('/') + '/' + self.outfile + '_simalign_argmax', 'w') as fo:
            start_time = datetime.datetime.now()
            for src_line, trg_line in zip(src, trg):
                self.ctr+=1
                src_sentence = src_line.strip().split()
                trg_sentence = trg_line.strip().split()

                alignments = self.model.get_word_aligns(src_sentence, trg_sentence)
                fo.write(utilities.simalign2pharaoh(str(alignments['inter'])) + '\n')
                now_time = datetime.datetime.now()
                if self.ctr % 100 == 0:
                    intervaltime = now_time - start_time
                    estimatedtime = "{0[0]}:{0[1]}:{0[2]}".format(days_hours_minutes(intervaltime, self.num_lines/self.ctr))
                    update_progress_notice(self.ctr, self.num_lines, start_time, estimatedtime)


class eflomal:
    def __init__(self, info_dict):
        self.generate_priors = info_dict['generate_priors']
        self.eflomal_folder = info_dict['TOOLS']['eflomal']
        self.fastalign_folder = info_dict['TOOLS']['fastalign']
        self.fa_folder = info_dict['FILE_FOLDERS']['fa']
        self.training_corpus_name = info_dict['DATA']['trainingcorpus']
        self.alignment_set_name = info_dict['DATA']['alignmentset']
        self.output_folder_alignments = info_dict['FILE_FOLDERS']['output']
        self.sym_types = info_dict['sym_types']
        self.alignment_files = []

    def align(self):
        if self.generate_priors:
            print('Training priors...')
            utilities.align_eflomal_makepriors(self.eflomal_folder, self.fa_folder, self.training_corpus_name, self.output_folder_alignments)

            # symmetrize and make priors
            for sym_type in self.sym_types:
                print('Symmetrizing...')
                utilities.align_eflomal_syms(self.fastalign_folder, self.eflomal_folder, self.fa_folder, self.training_corpus_name, self.output_folder_alignments, sym_type)

        for sym_type in self.sym_types + ['fr']:
            print('Aligning ', sym_type)
            utilities.align_eflomal_tests(self.fastalign_folder, self.eflomal_folder, self.fa_folder, sym_type, self.output_folder_alignments, self.training_corpus_name, self.alignment_set_name)
            sym_eflomal = self.sym_types + ['fwd', 'rev']

            for i in sym_eflomal:
                try:
                    self.alignment_files.append(self.output_folder_alignments + '/eflomal__' + self.alignment_set_name + '_' + self.training_corpus_name + '_____' + utilities.sym_type2short(sym_type) + '.' + utilities.sym_type2short(i))
                except Exception as e:
                    print(e)

class fastalign:
    def __init__(self, info_dict):
        self.generate_priors = info_dict['generate_priors']
        self.eflomal_folder = info_dict['TOOLS']['eflomal']
        self.fastalign_folder = info_dict['TOOLS']['fastalign']
        self.fa_folder = info_dict['FILE_FOLDERS']['fa']
        self.training_corpus_name = info_dict['DATA']['trainingcorpus']
        self.fa2align = info_dict['fa2align']
        self.alignment_set_name = info_dict['DATA']['alignmentset']
        self.output_folder_alignments = info_dict['FILE_FOLDERS']['output']
        self.sym_types = info_dict['sym_types']
        self.training_corpus = info_dict['training_corpus']
        self.testNumLines = info_dict['num_test_lines']
        self.alignment_files = []

    def align(self):
        print('Forward alignments...')
        fwd_file = utilities.align_fastalign(self.fastalign_folder, self.fa_folder, self.fa2align, self.output_folder_alignments, 'fwd', self.fa2align + '_fastalign')
        print('Reverse alignments...')
        rev_file = utilities.align_fastalign(self.fastalign_folder, self.fa_folder, self.fa2align, self.output_folder_alignments, 'rev', self.fa2align + '_fastalign')

        if self.training_corpus:
            utilities.decouple_testset(self.output_folder_alignments, fwd_file, self.testNumLines)
            utilities.decouple_testset(self.output_folder_alignments, rev_file, self.testNumLines)

        for sym_type in self.sym_types:
            self.alignment_files.append(utilities.symmetrize_alignments(self.fastalign_folder, fwd_file, rev_file, sym_type))
        self.alignment_files.append(fwd_file)
        self.alignment_files.append(rev_file)


class mgiza:
    def __init__(self, info_dict):
        self.generate_priors = info_dict['generate_priors']
        self.mgiza_folder = info_dict['TOOLS']['mgiza']
        self.fastalign_folder = info_dict['TOOLS']['fastalign']
        self.fa_folder = info_dict['FILE_FOLDERS']['fa']
        self.training_corpus_name = info_dict['DATA']['trainingcorpus']
        self.fa2align = info_dict['fa2align']
        self.alignment_set_name = info_dict['DATA']['alignmentset']
        self.output_folder_alignments = info_dict['FILE_FOLDERS']['output']
        self.sym_types = info_dict['sym_types']
        self.training_corpus = info_dict['training_corpus']
        self.testNumLines = info_dict['num_test_lines']
        self.align_file = info_dict['align_file']
        self.input_fa = info_dict['input_fa']
        self.n_value = info_dict['n_value']
        self.alignment_files = []

    def align(self):
        mgiza_utils = self.mgiza_folder + '/mgiza'
        mgiza_mkcls = self.mgiza_folder + '/mkcls'

        self.training_corpus_name = self.fa2align.rstrip('.fa')
        print('Concatenating train and test corpora')
        self.lines2align = self.fa_folder + '/' + self.training_corpus_name

        if self.input_fa:
            print('Converting from fa format')
            utilities.format_fa2giza(self.fa_folder, self.training_corpus_name)

        src_file = self.training_corpus_name + '.src'
        trg_file = self.training_corpus_name + '.trg'

        # count training lines
        print('Counting lines')
        trainNumLines = get_line_count(self.fa_folder + '/' + self.training_corpus_name + '.fa')
        print(str(trainNumLines) + ' in training set')

        print('mkcls source')
        utilities.align_mgiza_mkcls(self.mgiza_folder, self.n_value, self.fa_folder, self.training_corpus_name, 'src')
        print('mkcls target')
        utilities.align_mgiza_mkcls(self.mgiza_folder, self.n_value, self.fa_folder, self.training_corpus_name, 'trg')
        # býr til file1.vcb.classes, file1.vcb.classes.cats

        print('Aligning sentence files')
        utilities.align_mgiza_p2s(self.mgiza_folder, self.fa_folder, self.training_corpus_name)
        # býr til file1_file2.snt, file2_file1.snt, file1.vcb, file2.vcb

        print('creating coordination file')
        utilities.align_mgiza_cooc(self.mgiza_folder, self.fa_folder, self.training_corpus_name)
        # býr til file.cooc

        print('running mgiza')
        utilities.align_mgiza(self.mgiza_folder, self.fa_folder, self.training_corpus_name, self.output_folder_alignments, self.n_value)

        print('merging parts')
        utilities.align_mgiza_merge(self.mgiza_folder, self.output_folder_alignments, self.training_corpus_name)

        # formatera og decouple
        print('formatting output')
        self.alignment_files.append(utilities.format_giza2pharaoh_notest(self.output_folder_alignments, self.training_corpus_name))

        print('cleaning up')
        utilities.align_mgiza_clean(self.output_folder_alignments, self.training_corpus_name)

class gizapp:
    def __init__(self, info_dict):
        self.generate_priors = info_dict['generate_priors']
        self.gizapp_folder = info_dict['TOOLS']['gizapp']
        self.fastalign_folder = info_dict['TOOLS']['fastalign']
        self.fa_folder = info_dict['FILE_FOLDERS']['fa']
        self.training_corpus_name = info_dict['DATA']['trainingcorpus']
        self.fa2align = info_dict['fa2align']
        self.alignment_set_name = info_dict['DATA']['alignmentset']
        self.output_folder_alignments = info_dict['FILE_FOLDERS']['output']
        self.sym_types = info_dict['sym_types']
        self.training_corpus = info_dict['training_corpus']
        self.testNumLines = info_dict['num_test_lines']
        self.align_file = info_dict['align_file']
        self.input_fa = info_dict['input_fa']
        self.n_value = info_dict['n_value']
        self.alignment_files = []

    def align(self):
        gizapp_utils = self.gizapp_folder + '/GIZA++-v2'
        gizapp_mkcls = self.gizapp_folder + '/mkcls-v2'

        self.training_corpus_name = self.fa2align.strip('.fa')
        print('Concatenating train and test corpora')
        self.lines2align = self.fa_folder + '/' + self.training_corpus_name

        if self.input_fa:
            print('Converting from fa format')
            utilities.format_fa2giza(self.fa_folder, self.training_corpus_name)

        src_file = self.training_corpus_name + '.src'
        trg_file = self.training_corpus_name + '.trg'

        # count training lines
        print('Counting lines')
        trainNumLines = get_line_count(self.fa_folder + '/' + self.training_corpus_name + '.fa')
        print(str(trainNumLines) + ' in training set')

        print('Aligning sentence files')
        utilities.align_gizapp_p2s(gizapp_utils, self.fa_folder, self.training_corpus_name)
        # býr til file1_file2.snt, file2_file1.snt, file1.vcb, file2.vcb

        print('mkcls source')
        #utilities.align_gizapp_mkcls(gizapp_mkcls, self.n_value, self.fa_folder, self.training_corpus_name, 'src')
        print('mkcls target')
        utilities.align_gizapp_mkcls(gizapp_mkcls, self.n_value, self.fa_folder, self.training_corpus_name, 'trg')
        # býr til file1.vcb.classes, file1.vcb.classes.cats

        print('creating coordination file')
        utilities.align_gizapp_cooc(gizapp_utils, self.fa_folder, self.training_corpus_name)
        # býr til file.cooc

        print('running giza++')
        utilities.align_gizapp(gizapp_utils, self.fa_folder, self.training_corpus_name, self.output_folder_alignments)

        # formatera og decouple
        print('formatting output')
        self.alignment_files.append(utilities.format_giza2pharaoh_notest(self.output_folder_alignments, self.training_corpus_name))



class awesome:
    def __init__(self, info_dict):
        self.fa_folder = info_dict['FILE_FOLDERS']['fa']
        self.output_folder_alignments = info_dict['FILE_FOLDERS']['output']
        self.align_file = info_dict['align_file']
        self.model = info_dict['model']

    def align(self):
        utilities.align_awesome(self.fa_folder + '/' + self.align_file, self.output_folder_alignments + '/' + self.align_file + '.awesome', self.model)
