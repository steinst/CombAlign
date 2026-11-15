#!/bin/bash
# awesome-align fine-tuning (Dou & Neubig 2021) of LaBSE (AccAlign-style, layer 6) or mBERT (layer 8) on N sentence
# pairs of data/train/parice, then evaluation on the gold set with experiments/ft_eval.py.
#   bash experiments/finetune_awesome.sh labse|mbert [N=40000] [extra awesome-train args, e.g. --no_cuda]
set -e
cd "$(dirname "$0")/.."
which=$1; N=${2:-40000}; shift; shift || true
case $which in
  labse) base=setu4993/LaBSE; layer=6; thr=0.1; obj="--train_so" ;;
  mbert) base=bert-base-multilingual-cased; layer=8; thr=0.001; obj="--train_so --train_tlm" ;;
  *) echo "labse|mbert"; exit 1 ;;
esac
mkdir -p work/awesome
paste -d'\t' <(head -n $N data/train/parice.en) <(head -n $N data/train/parice.is) | sed 's/\t/ ||| /' > work/awesome/train.txt
for s in dev test; do
  paste -d'\t' <(cut -f2- data/gold/${s}_sentences.en) <(cut -f2- data/gold/${s}_sentences.is) | sed 's/\t/ ||| /' > work/awesome/$s.txt
done
awesome-train --model_name_or_path $base --train_data_file work/awesome/train.txt --output_dir work/awesome/${which}_ft \
  --do_train $obj --align_layer $layer --extraction softmax --softmax_threshold $thr --per_gpu_train_batch_size 16 \
  --gradient_accumulation_steps 2 --learning_rate 2e-5 --num_train_epochs 1 --save_steps 100000 --overwrite_output_dir "$@"
python experiments/ft_eval.py ${which}_ft
