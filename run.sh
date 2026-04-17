#!/usr/bin/env bash
python train.py --config config.yaml  2>&1 | tee train_llava_1_5.log
