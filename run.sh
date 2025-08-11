#/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 train_accelerate.py --config="./configs/speech2speech.yaml"
