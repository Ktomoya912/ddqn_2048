#!/bin/bash

source activate torch

# 引数が存在するかチェック 長さ3でない場合エラー
if [ "$#" -ne 2 ]; then
    echo "使用方法: $0 <seed> <type>"
    exit 1
fi

if [ "$2" = "1" ]; then
    CUDA_VISIBLE_DEVICES=0 nohup python src/learning-nn.py -s $1 -m CNN_DEEP --symmetry -H 120 &
    CUDA_VISIBLE_DEVICES=1 nohup python src/learning-toggle.py -s $1 -m CNN_DEEP --symmetry -H 120 --ddqn_type toggle &
elif [ "$2" = "2" ]; then
    CUDA_VISIBLE_DEVICES=0 nohup python src/learning-buffer.py -s $1 -m CNN_DEEP --symmetry -H 120 --target_update_freq 50 &
    CUDA_VISIBLE_DEVICES=1 nohup python src/learning.py -s $1 -m CNN_DEEP --symmetry -H 120 --target_update_freq 100 &
else
    echo "無効なタイプです。1または2を指定してください。"
    exit 1
fi
