r"""Code from https://github.com/AudioFans/audidata/blob/main/audidata/datasets/musdb18hq.py"""
from __future__ import annotations

from ast import expr_context
import os
from pathlib import Path
from typing import Union
from transformers import T5Tokenizer
import torch
import pandas as pd
from torchaudio.transforms import Resample

import librosa
import numpy as np
from datasets import load_dataset, load_from_disk, Audio
from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from torch.utils.data import Dataset
from typing_extensions import Literal
from torch.nn.utils.rnn import pad_sequence


class INSTRUCTION_SPEECH(Dataset):
    def __init__(
        self,
        root: str = "/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks", 
        split: Literal["train", "test"] = "train",
        sr: int = 16000,
        max_duration = 5,
        crop: callable = RandomCrop(clip_duration=10.),
        audio_base_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_train'
    ) -> None:
        r"""
        time_align: str. "strict" indicates all stems are aligned (from the 
            same song and have the same start time). "group" indictates 
            target stems / background stems are aligned. "random" indicates 
            all stems are from different songs with different start time.
        """

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.max_duration = max_duration
        self.audio_base_path = audio_base_path
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        # self.audio_list = load_dataset(
        #     'parquet',
        #     data_files={'train': [self.root]},
        #     split='train',
        # )
        self.audio_list = pd.read_csv(self.root).to_dict(orient='records')
        self.base_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_train'

    def pad_to_length(self, waveform: np.ndarray, target_len: int = 220500) -> np.ndarray:
        """
        waveform: np.ndarray of shape (1, T)
        target_len: target length in samples
        """
        current_len = waveform.shape[1]
        if current_len >= target_len:
            return waveform[:, :target_len]  # truncate if too long
        pad_width = target_len - current_len
        return np.pad(waveform, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)

    def pad_add_1s(self, waveform: np.ndarray) -> np.ndarray:
        """
        waveform: np.ndarray of shape (1, T)
        target len: add 22050 sample at the end of audio
        """
        return np.pad(waveform, ((0, 0), (0, 22050)), mode='constant', constant_values=0.0)

    def __getitem__(self, index) -> dict:
        N = len(self.audio_list)
        base_index = (index // 2) % N
        mirror = (index % 2 == 1)

        while True:
            try:
                item = self.audio_list[base_index]
            except Exception:
                __import__('remote_pdb').set_trace()

            if not mirror:
                style_instruction = item['trg_instruct']
                src_path = f"{self.base_path}/{item['audio_path']}"
                trg_path = item['vc_path']
                src_name = Path(src_path).stem
                trg_name = Path(trg_path).stem
            else:
                style_instruction = item['src_instruct']
                src_path = item['vc_path']
                trg_path = f"{self.base_path}/{item['audio_path']}"
                src_name = Path(src_path).stem
                trg_name = Path(trg_path).stem

            src_audio_duration = librosa.get_duration(path=src_path)
            trg_audio_duration = librosa.get_duration(path=trg_path)

            if src_audio_duration > self.max_duration or trg_audio_duration > self.max_duration:
                base_index = (base_index + 1) % N
                continue

            break  # 找到合格样本

        text = self.tokenizer(style_instruction, return_tensors='pt')
        src_audio = load(path=src_path, sr=self.sr)
        trg_audio = self.pad_add_1s(load(path=trg_path, sr=self.sr))

        data = {
            "dataset_name": "INSTRUCTION_SPEECH",
            "source": src_audio,
            "target": trg_audio,
            "instruction": text,
            "src_name": src_name,
            "trg_name": trg_name,
            "src_valid_length": src_audio.shape[1],
            "trg_valid_length": trg_audio.shape[1],
        }
       
        return data


    def __len__(self) -> int:
        return len(self.audio_list) * 2


def collate_fn(batch):
    input_ids_list = [item['instruction']['input_ids'][0] for item in batch]  # shape: [seq_len]
    attention_mask_list = [item['instruction']['attention_mask'][0] for item in batch]
    # Pad input_ids 和 attention_mask
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    src_valid_length_list = [item['src_valid_length'] for item in batch]
    trg_valid_length_list = [item['trg_valid_length'] for item in batch]
    # source_audio = pad_sequence([torch.from_numpy(item['source']) for item in batch], batch_first=True, padding_value=0)
    # target_audio = pad_sequence([torch.from_numpy(item['target']) for item in batch], batch_first=True, padding_value=0)

    return {
        'source': [torch.from_numpy(item['source']) for item in batch],
        'target': [torch.from_numpy(item['target']) for item in batch],
        'instruction': {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
        },
        'src_name': [item['src_name'] for item in batch],
        'trg_name': [item['trg_name'] for item in batch],
    }
