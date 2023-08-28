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
