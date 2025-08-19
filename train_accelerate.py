from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Literal
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
import matplotlib.pyplot as plt
import soundfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm
from transformers import T5EncoderModel
from audio_flow.encoders.bigvgan import Mel_BigVGAN_22kHz
import torch.nn.functional as F
import wandb
from audio_flow.utils import LinearWarmUp, Logmel, parse_yaml, requires_grad, update_ema
from train import (get_dataset, get_sampler, get_model, get_data_transform, 
    get_optimizer_and_scheduler, validate)
from audio_flow.datasets.instruction_speech import collate_fn


global_rank = int(os.environ.get("RANK", 0))  # 全局rank
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前机器的rank
is_main_process = global_rank == 0 and local_rank == 0


def pad_tensors(tensor_list, pad_value=0):
    """
    把 list[tensor] pad 成相同长度，并返回 pad 后的张量和每个样本 pad 的个数
    Args:
        tensor_list: list[Tensor]，每个 tensor 的 shape 形如 (seq_len, dim) 或 (seq_len,)
        pad_value: 用于填充的值
    Returns:
        padded: Tensor, shape = (batch, max_len, ...)
        pad_counts: Tensor, shape = (batch,)
    """
    lengths = torch.tensor([t.size(2) for t in tensor_list])
    max_len = lengths.max().item()

    padded_list = []
    valid_length = []
    for t in tensor_list:
        pad_len = max_len - t.size(2)
        valid_length.append(t.shape[2])
        # 左右 pad 规则 (pad_left, pad_right, pad_top, pad_bottom, ...)
        padded_list.append(F.pad(t, (0, 0, 0, pad_len), value=pad_value))

    padded = torch.cat(padded_list, dim=0)
    valid_length = torch.tensor(valid_length)
    return padded, valid_length


def train(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    print(configs)
    # 得到当天的日期 按照0808的格式
    import time
    date = time.strftime("%m%d", time.localtime())
    # Checkpoints directory
    config_name = Path(config_path).stem
    save_dir =Path(f"{configs['save_dir']}/{args.exp_name}")
    ckpts_dir = Path(save_dir, "checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train", mode="train")

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Data processor
    data_transform = get_data_transform(configs)

    # Flow matching data processor
    fm = ConditionalFlowMatcher(sigma=0.)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    )

    # EMA (optional)
    ema = deepcopy(model)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Prepare for acceleration
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=configs["train"]["precision"],
        kwargs_handlers=[ddp_kwargs],
    )

    vocoder = Mel_BigVGAN_22kHz().to(accelerator.device)
    text_encoder = T5EncoderModel.from_pretrained('/mnt/bn/tanman-yg/chenqi/code/Speech_flow/pretrained/t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1').to(accelerator.device)
    vocoder.requires_grad_(False)
    text_encoder.requires_grad_(False)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    ema.to(accelerator.device)
    # Logger
    if wandb_log and is_main_process:
        wandb.init(project="audio_flow", name=f"{date}_{Path(save_dir).stem}")

    # Train
    total_steps = configs["train"]["training_steps"]  # 总步数
    progress_bar = tqdm(train_dataloader, total=total_steps, desc="Training", unit="iters")
    for step, data in enumerate(progress_bar):
        # ------ 1. Data preparation ------
        # 1.1 Transform data into latent representations and conditions
        # target, cond_dict, _ = data_transform(data)
        def extract_features(data):
            src_audios = []
            trg_audios = []
            with torch.no_grad():
                src_stereo = data["source"]  # (b, 1, l)
                for stereo in src_stereo:
                    source = vocoder.encode(stereo.to(accelerator.device).unsqueeze(0))  # (b, 1, t, f)
                    src_audios.append(source)
                trg_stereo = data["target"]  # (b, 1, l)
                for stereo in trg_stereo:
                    target = vocoder.encode(stereo.to(accelerator.device).unsqueeze(0))  # (b, 1, t, f)
                    trg_audios.append(target)
                # target = vocoder.encode(trg_stereo)  # (b, 1, t, f)
                src_audios, src_valid_length = pad_tensors(src_audios)
                trg_audios, trg_valid_length = pad_tensors(trg_audios)


                # Instruction as condition
                instruction = {k: v.to(accelerator.device) for k, v in data["instruction"].items()}
                outputs = text_encoder(**instruction)
                text_emb = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
                text_emb = text_emb.to(accelerator.device)
                sentence_embedding = text_emb.mean(dim=1)  # shape: (batch_size, hidden_size)

                cond_dict = {
                    "y": None,                        # one-hot
                    "c": sentence_embedding,          # global condition, [D]
                    "ct": None,                       # temporal condition, [D, T]
                    "ctf": None,                      # temporal-frequency condition, [D, T, F]
                    "cx": None,                       # cross-attention condition, [D, T, F]
                    'cld': None,                      # language condition, [D, l, hidden_size], mean l is the length of text feature, and hidden size is the dimention
                    "src_audio": src_audios,                      # audio condition, [D, C, T, F]
                }

                cond_sources = {
                    "src_audio": src_stereo, 
                    "trg_audio": trg_stereo
                }
            return trg_audios, cond_dict, cond_sources, src_valid_length, trg_valid_length
        
        with accelerator.accumulate(model):
            target, cond_dict, _, src_valid_length, trg_valid_length = extract_features(data)
            cond_dict['src_valid_length'] = src_valid_length.to(accelerator.device)
            cond_dict['trg_valid_length'] = trg_valid_length.to(accelerator.device)
            
            # 1.2 Noise
            noise = torch.randn_like(target)

            # 1.3 Get input and velocity
            t, xt, ut = fm.sample_location_and_conditional_flow(x0=noise, x1=target)

            # ------ 2. Training ------
            # 2.1 Forward
            model.train()
            vt = model(t=t, x=xt, cond_dict=cond_dict)
            # __import__('ipdb').set_trace()

            # 2.2 Loss
            loss_mask = torch.zeros_like(vt)
            for i, length in enumerate(cond_dict['trg_valid_length']):
                length_int = int(length.item())  # 转成 Python int
                loss_mask[i, :, :length_int] = 1
            loss = (((vt - ut) ** 2) * loss_mask).sum() / (loss_mask.sum() + 1e-8)

            # 2.3 Optimize
            optimizer.zero_grad()  # Reset all parameter.grad to 0
            accelerator.backward(loss)  # Update all parameter.grad
            optimizer.step()  # Update all parameters based on all parameter.grad
            update_ema(ema_model=ema, model=accelerator.unwrap_model(model))

            lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}", "average_len": f"{loss_mask.sum():.0f}/{loss_mask.numel():.0f}"})

        # 2.4 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0 and is_main_process:
            print("train loss: {:.4f}".format(loss.item()))

        # ------ 3. Evaluation ------
        # 3.1 Evaluate

        def latent_to_audio(x: Tensor) -> Tensor:
            r"""Ues vocoder to convert mel spectrogram to audio.

            Args:
                x: (b, c, t, f)

            Outputs:
                y: (b, c, l)
            """
            return vocoder.decode(x)


        if step % configs["train"]["test_every_n_steps"] == 0 and is_main_process:
            for split in ["train", "test"]:
                validate(
                    configs=configs,
                    data_transform=extract_features,
                    latent_to_audio=latent_to_audio,
                    model=accelerator.unwrap_model(model),
                    split=split,
                    out_dir=Path(save_dir, filename, config_name, f"steps={step}")
                )
            for split in ["train", "test"]:
                validate(
                    configs=configs,
                    data_transform=extract_features,
                    latent_to_audio=latent_to_audio,
                    model=accelerator.unwrap_model(ema),
                    split=split,
                    out_dir=Path(save_dir, filename, config_name, f"steps={step}_ema")
                )

        # 3.2 Log
        if step % configs["train"]["record_every_n_steps"] == 0 and is_main_process:
            if wandb_log:
                wandb.log(
                    data={
                        "train_loss": loss.item()
                    },
                    step=step
                )
        
        # 3.3 Save model
        if step % configs["train"]["save_every_n_steps"] == 0 and is_main_process:
            
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

            ckpt_path = Path(ckpts_dir, "step={}_ema.pth".format(step))
            torch.save(accelerator.unwrap_model(ema).state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break

        step += 1
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--exp_name", type=str, default="")
    args = parser.parse_args()

    train(args)