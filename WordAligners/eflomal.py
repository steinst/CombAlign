import datetime
from itertools import cycle

spinner = cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')

def update_progress_notice(ctr, num_lines, start_time, estimatedtime):
    now_time = datetime.datetime.now()
    print(" ", next(spinner), "{}/{}".format(str(ctr), str(num_lines)),
          ("  {}/{}".format(str(now_time - start_time), estimatedtime) if ctr % 100 == 0 else ""), end='\r')

def days_hours_minutes(td, multiple=1):
    td = td*multiple
    return str(td.seconds//3600 + td.days*24).zfill(2), str((td.seconds//60)%60).zfill(2), str(td.seconds%60).zfill(2)

class Eflomal:
    if args.generate_priors:
        # make first priors
        print('Training priors...')
        utilities.align_eflomal_makepriors(eflomal_folder, fa_folder, training_corpus_name, output_folder_alignments)

        # symmetrize and make priors
        for sym_type in sym_types:
            print('Symmetrizing...')
            utilities.align_eflomal_syms(fastalign_folder, eflomal_folder, fa_folder, training_corpus_name, output_folder_alignments, sym_type)
    for sym_type in sym_types + ['fr']:
        print('Aligning ', sym_type)
        utilities.align_eflomal_tests(fastalign_folder, eflomal_folder, fa_folder, sym_type, output_folder_alignments, training_corpus_name, alignment_set_name)
        sym_eflomal = sym_types + ['fwd', 'rev']

        for i in sym_eflomal:
            try:
                alignment_files.append(output_folder_alignments + '/eflomal__' + alignment_set_name + '_' + training_corpus_name + '_____' + utilities.sym_type2short(sym_type) + '.' + utilities.sym_type2short(i))
            except Exception as e:
                print(e)
