#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PYTHONPATH:/home/zouyou/workspaces/ASR/newKaldi/icefall/
export CUDA_VISIBLE_DEVICES="5"
export LANG="en_US.UTF-8"

set -eou pipefail

nj=8
stage=0
stop_stage=100

# Split L subset to this number of pieces
# This is to avoid OOM during feature extraction.
num_splits=100


dl_dir=$PWD/KeSpeech/KeSpeech
lang_char_dir=data/lang_char

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

# step 1：从 KeSpeech/Taskes/ASR文件中，生成recodings（记录音频信息）和supervisions（对应音频的text文件）文件
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare KeSpeech manifest"

  if [ ! -f data/manifests/.manifest_load_complete ]; then
    mkdir -p data/manifests
    # lhotse prepare wenet-speech $dl_dir/WenetSpeech data/manifests -j $nj
    python3 ./lhotse/ke_speech.py \
      --corpus-dir $dl_dir \
      --output-dir data/manifests \
      --num-jobs $nj
    touch data/manifests/.manifest_load_complete
  fi
fi


# step 2：从manifests中的xx.jsonl.gz文件生成fbank/cuts_xx_raw.jsonl.gz
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Preprocess KeSpeech manifest"
  if [ ! -f data/fbank/.preprocess_complete ]; then
    mkdir -p data/manifests
    python3 ./local/preprocess_kespeech.py
    touch data/fbank/.preprocess_complete
  fi
fi

# step 3：用fbank/cuts_DEV/TEST_raw.jsonl.gz 生成 fbank/cuts_DEV/TEST.jsonl.gz
#         用fbank/cuts_DEV/TEST.jsonl.gz 和 audio文件 生成 fbank/feats_DEV/TEST.h5（fbank特征）
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute features for dev_phase1, dev_phase2 and test subsets of KeSpeech (may take 2 minutes)"
  python3 ./local/compute_fbank_kespeech_dev_test.py
fi



# step 4：将fbank/cuts_train_phase1_raw.jsonl.gz 切分成100份fbank/train_phase1_split_100/cuts_train_phase1_raw.xxxx.jsonl.gz 
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Split train_phase1 subset into ${num_splits} pieces"
  split_dir=data/fbank/train_phase1_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split $num_splits ./data/fbank/cuts_train_phase1_raw.jsonl.gz $split_dir
    touch $split_dir/.split_completed
  fi
fi


# step 5：将fbank/cuts_train_phase2_raw.jsonl.gz 切分成100份fbank/train_phase2_split_100/cuts_train_phase2_raw.xxxx.jsonl.gz 
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Split train_phase2 subset into ${num_splits} pieces"
  split_dir=data/fbank/train_phase2_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split $num_splits ./data/fbank/cuts_train_phase2_raw.jsonl.gz $split_dir
    touch $split_dir/.split_completed
  fi
fi



# step 6：用fbank/train_phase1_split_100/cuts_train_phase1_raw.xxxx.jsonl.gz 生成 fbank/train_phase1_split_100/cuts_train_phase1.xxxx.jsonl.gz
#         用fbank/train_phase1_split_100/cuts_train_phase1.xxxx.jsonl.gz 和 audio文件 生成 fbank/train_phase1_split_100/feats_train_phase1_xxxx.lca（fbank特征）
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Compute features for train_phase1"
  python3 ./local/compute_fbank_kespeech_splits.py \
    --training-subset train_phase1 \
    --num-workers 20 \
    --batch-duration 600 \
    --start 0 \
    --num-splits $num_splits
fi


# step 7：用fbank/train_phase2_split_100/cuts_train_phase2_raw.xxxx.jsonl.gz 生成 fbank/train_phase2_split_100/cuts_train_phase2.xxxx.jsonl.gz
#         用fbank/train_phase2_split_100/cuts_train_phase2.xxxx.jsonl.gz 和 audio文件 生成 fbank/train_phase2_split_100/feats_train_phase2_xxxx.lca（fbank特征）
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Compute features for train_phase2"
  python3 ./local/compute_fbank_kespeech_splits.py \
    --training-subset train_phase2 \
    --num-workers 20 \
    --batch-duration 600 \
    --start 0 \
    --num-splits $num_splits
fi



# step 8: 将 fbank/train_phase1_split_100/cuts_train_phase1.xxxx.jsonl.gz 合并成 fbank/cuts_train_phase1.jsonl.gz
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Combine features for train_phase1"
  if [ ! -f data/fbank/cuts_train_phase1.jsonl.gz ]; then
    pieces=$(find data/fbank/train_phase1_split_100 -name "cuts_train_phase1.*.jsonl.gz")
    lhotse combine $pieces data/fbank/cuts_train_phase1.jsonl.gz
  fi
fi


# step 9: 将 fbank/train_phase2_split_100/cuts_train_phase2.xxxx.jsonl.gz 合并成 fbank/cuts_train_phase2.jsonl.gz
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Combine features for train_phase2"
  if [ ! -f data/fbank/cuts_train_phase2.jsonl.gz ]; then
    pieces=$(find data/fbank/train_phase2_split_100 -name "cuts_train_phase2.*.jsonl.gz")
    lhotse combine $pieces data/fbank/cuts_train_phase2.jsonl.gz
  fi
fi


# step 14: 生成fbank/musan_feats
if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
  log "Stage 14: Compute fbank for musan"
  mkdir -p data/fbank
  ./local/compute_fbank_musan.py
fi

########## 这里开始都是构建语言模型文件 ##########
# step 15-1: 通过 wenetspeech_supervisions_S.jsonl.gz 生成 lang_char/text文件
if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
  log "Stage 15: Prepare char based lang"
  mkdir -p $lang_char_dir

  if ! which jq; then
      echo "This script is intended to be used with jq but you have not installed jq
      Note: in Linux, you can install jq with the following command:
      1. wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
      2. chmod +x ./jq
      3. cp jq /usr/bin" && exit 1
  fi
  if [ ! -f $lang_char_dir/text ] || [ ! -s $lang_char_dir/text ]; then
    log "Prepare text."
    gunzip -c data/manifests/wenetspeech_supervisions_L.jsonl.gz \
      | jq '.text' | sed 's/"//g' \
      | ./local/text2token.py -t "char" > $lang_char_dir/text
  fi

  # The implementation of chinese word segmentation for text,
  # and it will take about 15 minutes.
  # step 15-2: 将lang_char/text文件分词，生成 lang_char/text_words_segmentation
  if [ ! -f $lang_char_dir/text_words_segmentation ]; then
    python3 ./local/text2segments.py \
      --num-process $nj \
      --input-file $lang_char_dir/text \
      --output-file $lang_char_dir/text_words_segmentation
  fi

  # step 15-3: 将分词文件lang_char/text_words_segmentation重新整理排序，生成lang_char/words_no_ids.txt（每个词一行）
  cat $lang_char_dir/text_words_segmentation | sed 's/ /\n/g' \
    | sort -u | sed '/^$/d' | uniq > $lang_char_dir/words_no_ids.txt

  # step 15-4: 将lang_char/words_no_ids.txt加上序号，生成lang_char/words.txt
  if [ ! -f $lang_char_dir/words.txt ]; then
    python3 ./local/prepare_words.py \
      --input-file $lang_char_dir/words_no_ids.txt \
      --output-file $lang_char_dir/words.txt
  fi
fi

# step 16: lang_char/text -> lang_char/token.txt
#          lang_char/token.txt & lang_char/words.txt -> lang_char/lexicon.txt
#          lang_char/lexicon.txt -> lexicon_disambig.txt
#          lexicon.txt, 转到 k2 里面的 fsa, 然后用 torch.save() 保存下来，就是 L.pt（这里的L指的是lexicon)
#          同理, L_disambig.pt 是由 lexicon_disambig.txt 生成的
#          lexicon.txt 与 lexicon_disambig.txt 的区别在于，lexicon_disambig.txt 中包含了 #1, #2, #3 等 disambig symbols.
if [ $stage -le 16 ] && [ $stop_stage -ge 16 ]; then
  log "Stage 16: Prepare char based L_disambig.pt"
  if [ ! -f data/lang_char/L_disambig.pt ]; then
    python3 ./local/prepare_char.py \
      --lang-dir data/lang_char
  fi
fi

# step 17: 生成lang_char/3-gram.unpruned.arpa 和 lm/
# If you don't want to use LG for decoding, the following steps are not necessary.
if [ $stage -le 17 ] && [ $stop_stage -ge 17 ]; then
  log "Stage 17: Prepare G"
  # It will take about 20 minutes.
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm
  if [ ! -f $lang_char_dir/3-gram.unpruned.arpa ]; then
    python3 ./shared/make_kn_lm.py \
      -ngram-order 3 \
      -text $lang_char_dir/text_words_segmentation \
      -lm $lang_char_dir/3-gram.unpruned.arpa
  fi

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building LG
    python3 -m kaldilm \
      --read-symbol-table="$lang_char_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $lang_char_dir/3-gram.unpruned.arpa > data/lm/G_3_gram.fst.txt
  fi
fi

# step 18: 生成lang_char/LG.pt
if [ $stage -le 18 ] && [ $stop_stage -ge 18 ]; then
  log "Stage 18: Compile LG"
  python ./local/compile_lg.py --lang-dir $lang_char_dir
fi
