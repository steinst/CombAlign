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
    def __init__(self, src_lines_file, trg_lines_file, outfile, model="xlmr", token_type="word", matching_methods="a", device="cuda"):
        self.model = SentenceAligner(model=model, token_type=token_type, matching_methods=matching_methods, device=device)
        self.num_lines = sum(1 for _ in open(src_lines_file))
        self.src_file = src_lines_file
        self.trg_file = trg_lines_file
        self.out_file = outfile
        self.matching_methods = matching_methods
        self.ctr = 0

    def align(self):
        with open(self.src_file) as src, open(self.trg_file) as trg, open('simalign_argmax_' + self.outfile, 'w') as fo:
            start_time = datetime.datetime.now()
            for src_line, trg_line in zip(src, trg):
                self.ctr+=1
                src_sentence = src_line.strip().split()
                trg_sentence = trg_line.strip().split()

                alignments = self.model.get_word_aligns(src_sentence, trg_sentence)
                fo.write(str(alignments['inter']) + '\n')
                now_time = datetime.datetime.now()
                if self.ctr % 100 == 0:
                    intervaltime = now_time - start_time
                    estimatedtime = "{0[0]}:{0[1]}:{0[2]}".format(days_hours_minutes(intervaltime,self.num_lines/self.ctr))
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
        self.sym_types = info_dict['matching_methods']
        self.alignment_files = []

    def align(self):
        if self.generate_priors:
            print('Training priors...')
            utilities.align_eflomal_makepriors(self.eflomal_folder, self.fa_folder, self.training_corpus_name, self.output_folder_alignments)

            # symmetrize and make priors
            for sym_type in self.sym_types:
                print('Symmetrizing...')
                utilities.align_eflomal_syms(self.fastalign_folder, self.eflomal_folder, self.fa_folder, self.training_corpus_name, self.output_folder_alignments, self.sym_type)

        for sym_type in self.sym_types + ['fr']:
            print('Aligning ', sym_type)
            utilities.align_eflomal_tests(self.fastalign_folder, self.eflomal_folder, self.fa_folder, self.sym_type, self.output_folder_alignments, self.training_corpus_name, self.alignment_set_name)
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
        self.sym_types = info_dict['matching_methods']
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
            self.alignment_files.append(utilities.symmetrize_alignments(self.fastalign_folder, self.output_folder_alignments, self.alignment_set_name, fwd_file, rev_file, sym_type, 'fastalign'))
        self.alignment_files.append(fwd_file)
        self.alignment_files.append(rev_file)


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
        self.sym_types = info_dict['matching_methods']
        self.training_corpus = info_dict['training_corpus']
        self.testNumLines = info_dict['num_test_lines']
        self.align_file = info_dict['align_file']
        self.input_fa = info_dict['input_fa']
        self.alignment_files = []

    def align(self):
        gizapp_utils = self.gizapp_folder + '/GIZA++-v2'
        gizapp_mkcls = self.gizapp_folder + '/mkcls-v2'

        if self.align_file:
            training_corpus_name = self.alignment_set_name
            print('Concatenating train and test corpora')
            lines2align = self.fa_folder + '/' + self.training_corpus_name + '.fa'

        if self.input_fa:
            import format_fa2giza
            print('Converting from fa format')
            format_fa2giza.convert(self.fa_folder, self.training_corpus_name.strip('.fa'))

        src_file = self.training_corpus_name + '.src'
        trg_file = self.training_corpus_name + '.trg'

        # count training lines
        print('Counting lines')
        trainNumLines = get_line_count(self.fa_folder + '/' + self.training_corpus_name + '.fa')
        print(str(trainNumLines) + ' in training set')

        print('Aligning sentence files')
        utilities.align_gizapp_p2s(gizapp_utils, self.fa_folder, self.corpus_file)
        # býr til file1_file2.snt, file2_file1.snt, file1.vcb, file2.vcb

        print('mkcls source')
        #utilities.align_gizapp_mkcls(gizapp_mkcls, args.n_value, fa_folder, corpus_file, 'src')
        print('mkcls target')
        #utilities.align_gizapp_mkcls(gizapp_mkcls, args.n_value, fa_folder, corpus_file, 'trg')
        # býr til file1.vcb.classes, file1.vcb.classes.cats

        print('creating coordination file')
        #utilities.align_gizapp_cooc(gizapp_utils, fa_folder, corpus_file)
        # býr til file.cooc

        print('running giza++')
        utilities.align_gizapp(gizapp_utils, self.fa_folder, self.training_corpus_name.strip('.fa'), self.output_folder_alignments)

        # formatera og decouple
        print('formatting output')
        self.alignment_files.append(utilities.format_giza2pharaoh_notest(self.output_folder_alignments, self.training_corpus_name.strip('.fa')))



#class Awesome: