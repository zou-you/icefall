#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/home/zouyou/workspaces/ASR/newKaldi/lhotse:$PYTHONPATH
export PYTHONPATH=/home/zouyou/workspaces/ASR/newKaldi/icefall:$PYTHONPATH

set -eou pipefail

stage=-1
stop_stage=100

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: Download Pre-trained model"
  
  # clone from huggingface
  git lfs install
  git clone git@hf.co:pfluo/k2fsa-zipformer-chinese-english-mixed

fi


log "Dataset: Self-Data"
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare self_data manifest"

  if [ ! -f data/manifests/.self_data.done ]; then
    mkdir -p data/manifests
    python local/generate_manifest_self_data.py --corpus-dir $dl_dir/self_data --output-dir data/manifests
    touch data/manifests/.self_data.done
  fi
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Compute fbank for self_data"
  if [ ! -f data/fbank/.self_data.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_self_data.py
    touch data/fbank/.self_data.done
  fi
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare AISHELL-1"
  if [ -e ../../aishell/ASR/data/fbank/.aishell.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_train) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_dev) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_test) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_train.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_dev.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_test.jsonl.gz) .
    cd ../..
  else
    log "Abort! Please run ../../aishell/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Start fine-tuning"
  
  # The following configuration of lr schedule should work well
  # You may also tune the following parameters to adjust learning rate schedule
  base_lr=0.0001
  lr_epochs=100
  lr_batches=100000

  # We recommend to start from an averaged model
  finetune_ckpt=pretrainedmodel/k2fsa-zipformer-chinese-english-mixed/exp/pretrained.pt
  lang_dir=pretrainedmodel/k2fsa-zipformer-chinese-english-mixed/data/lang_char_bpe
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

  ./pruned_transducer_stateless7_streaming/finetune.py \
    --world-size 6 \
    --master-port 18180 \
    --num-epochs 20 \
    --start-epoch 1 \
    --exp-dir pruned_transducer_stateless7_streaming/exp_aishell_and_selfdata_finetune \
    --feedforward-dims "1024,1024,1536,1536,1024" \
    --base-lr $base_lr \
    --lr-epochs $lr_epochs \
    --lr-batches $lr_batches \
    --lang-dir $lang_dir \
    --bpe-model pretrainedmodel/k2fsa-zipformer-chinese-english-mixed/data/lang_char_bpe/bpe.model \
    --do-finetune True \
    --enable-musan False \
    --finetune-ckpt $finetune_ckpt \
    --max-duration 200
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Decoding"

  epoch=15
  avg=10
  lang_dir=pretrainedmodel/k2fsa-zipformer-chinese-english-mixed/data/lang_char_bpe

  for m in greedy_search modified_beam_search; do
    python pruned_transducer_stateless7_streaming/finetune_decode.py \
    --epoch $epoch \
    --avg $avg \
    --use-averaged-model True \
    --beam-size 4 \
    --feedforward-dims "1024,1024,1536,1536,1024" \
    --exp-dir pruned_transducer_stateless7_streaming/exp_aishell_and_selfdata_finetune \
    --bpe-model pretrainedmodel/k2fsa-zipformer-chinese-english-mixed/data/lang_char_bpe/bpe.model \
    --lang-dir $lang_dir \
    --max-duration 400 \
    --decoding-method $m
  done
fi


if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: EXPORT-ONNX"
  ./pruned_transducer_stateless7_streaming/export-onnx-zh.py \
    --tokens pretrainedmodel/k2fsa-zipformer-chinese-english-mixed/data/lang_char_bpe/tokens.txt \
    --use-averaged-model 0 \
    --epoch 20 \
    --avg 1 \
    --exp-dir pruned_transducer_stateless7_streaming/exp_aishell_and_selfdata_finetune \
    --decode-chunk-len 32 \
    --num-encoder-layers "2,4,3,2,4" \
    --feedforward-dims "1024,1024,1536,1536,1024" \
    --nhead "8,8,8,8,8" \
    --encoder-dims "384,384,384,384,384" \
    --attention-dims "192,192,192,192,192" \
    --encoder-unmasked-dims "256,256,256,256,256" \
    --zipformer-downsampling-factors "1,2,4,8,2" \
    --cnn-module-kernels "31,31,31,31,31" \
    --decoder-dim 512 \
    --joiner-dim 512
fi
