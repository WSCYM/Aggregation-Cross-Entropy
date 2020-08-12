#!/usr/bin/env bash
filename="../log/log/log_`date +%y_%m_%d_%H_%M_%S`.txt"
CUDA_VISIBLE_DEVICES=1 python -u main.py \
  2>&1 | tee $filename