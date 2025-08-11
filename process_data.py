from datasets import load_dataset, load_from_disk, Audio
import os

parquet_path = "/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks/**"  # 或者通配符列表
arrow_out = "/mnt/fast/nobackup/scratch4weeks/yc01815/data_cache"          # 建议放到本机SSD

use_cols = ["src_audio", "trg_audio", "src_instruct", "trg_instruct"]  # 按需改
target_sr = 16000
num_proc = 8  # 机器越强越大

use_cols = ["src_audio", "trg_audio", "src_instruct", "trg_instruct"]  # 按需改

ds = load_dataset(
    "parquet",
    data_files={"train": [parquet_path]},
    split="train"
)

# 裁掉多余列
ds = ds.select_columns([c for c in use_cols if c in ds.column_names])

# 保存为 Arrow 分片（mmap 读取快很多）
os.makedirs(arrow_out, exist_ok=True)
ds.save_to_disk(arrow_out)
print(f"Saved to {arrow_out}")