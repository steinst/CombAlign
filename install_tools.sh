#fastalign

sudo apt-get install libgoogle-perftools-dev libsparsehash-dev -y

cd aligners
git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
cd ../..

#eflomal
git clone https://github.com/robertostling/eflomal.git
cd eflomal
make
sudo make install
python3 setup.py install
cd ..

#awesome

#mgiza
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
cmake .
make
make install
cd ..

#giza-pp
git clone https://github.com/moses-smt/giza-pp.git
cd giza-pp
make
cd ../..

cd ..
mkdir -p aligndata
cd aligndata
mkdir -p training_corpus
mkdir -p tokenized_sentences
mkdir -p alignments
mkdir -p fa_files
cd fa_files
touch empty.fa
cd ../../CombAlign
