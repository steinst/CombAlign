create a conda environment:
conda create --name combalign python=3.8
conda activate combalign

git clone https://github.com/steinst/CombAlign.git
cd CombAlign

pip install -r requirements.txt
bash install_tools.sh

this assumes Boost is installed 

Alignment systems used:

eflomal

bæta við tokenize


fast-align
==========
Follow instructions on https://github.com/clab/fast_align

sudo apt-get install libgoogle-perftools-dev libsparsehash-dev cmake
git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make

Add the fast_align/build folder to config.ini [TOOLS]->FastAlign
[skrifta til þess]

If you use fast_align, please cite the paper/papers listed on https://github.com/clab/fast_align.


```bash
cd ~/CombAlign
python word_alignment.py -m fastalign -fa -align splits_lem
python word_alignment.py -m eflomal -ifa -gp  -align splits_lem
python word_alignment.py -m simalign --model bert -align split_aa_af --device cuda
python word_alignment.py -m simalign --model xlmr -align split_aa_af --device cuda
python word_alignment.py -m awesome -fa -align split_aa_af --model bert-base-multilingual-cased -dno 1
python word_alignment.py -m gizapp -ifa -align splits_lem

python combi_alignment_no_test.py -m unionmin3 -o sentalign_final_filter.combialign -dir /home/steinst/aligndata/alignments -a sentalign_final_filter_lem.pharaoh eflomal__sentalign_final_filter_lem_empty_____intersect.intersect sentalign_final_filter_tok_simalign_bert_argmax sentalign_final_filter_tok_simalign_xlmr_argmax sentalign_final_filter_tok.awesome sentalign_final_filter_lem.fa_fastalign.intersect
```