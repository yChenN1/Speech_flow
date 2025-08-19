#/bin/bash
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,127.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16
export HF_DATASETS_CACHE=/mnt/bn/tanman-yg/chenqi/datas/.hf_dataset_cache

# 获取当前日期（MMDD）
TODAY=$(date +%m%d)
EXP_NAME=$1

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 train_accelerate.py --config="./configs/speech2speech.yaml"
# accelerate launch --multigpu --num_processes 2 train_accelerate.py --config="./configs/speech2speech.yaml"

accelerate launch \
    --num_processes 8 \
    --dynamo_backend "no" \
    --main_process_port ${PORT} \
    --mixed_precision bf16 \
    train_accelerate.py --config="./configs/speech2speech.yaml" --exp_name ${EXP_NAME}\
    2>&1 | tee ${TODAY}_a2a_${EXP_NAME}.log