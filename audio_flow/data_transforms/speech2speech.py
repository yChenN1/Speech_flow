from tkinter import NO
import torch
import torch.nn as nn
from torch import Tensor
from transformers import T5EncoderModel
from audio_flow.encoders.bigvgan import Mel_BigVGAN_22kHz


class Speech2Speech_Mel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vocoder = Mel_BigVGAN_22kHz()
        self.text_encoder = T5EncoderModel.from_pretrained('t5-base')

    def __call__(self, data: dict) -> tuple[Tensor, dict, dict]:
        r"""Transform data into latent representations and conditions.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: mel bins
        """        
        device = next(self.parameters()).device

        # Mel spectrogram target
        src_stereo = data["source"].to(device)  # (b, 1, l)
        source = self.vocoder.encode(src_stereo)  # (b, 1, t, f)
        
        trg_stereo = data["target"].to(device)  # (b, 1, l)
        target = self.vocoder.encode(trg_stereo)  # (b, 1, t, f)

        # Instruction as condition
        instruction = {k: v.to(device) for k, v in data["instruction"].items()}
        with torch.no_grad():
            self.text_encoder.eval()
            outputs = self.text_encoder(**instruction)
            
        text_emb = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        text_emb = text_emb.to(device)
        sentence_embedding = text_emb.mean(dim=1)  # shape: (batch_size, hidden_size)


        cond_dict = {
            "y": None,                        # one-hot
            "c": sentence_embedding,          # global condition, [D]
            "ct": None,                       # temporal condition, [D, T]
            "ctf": source,                    # temporal-frequency condition, [D, T, F]
            "cx": None,                       # cross-attention condition, [D, T, F]
            'cld': None                       # language condition, [D, l, hidden_size], mean l is the length of text feature, and hidden size is the dimention
        }

        cond_sources = {
            "src_audio": src_stereo, 
            "trg_audio": trg_stereo
        }

        return target, cond_dict, cond_sources



    def latent_to_audio(self, x: Tensor) -> Tensor:
        r"""Ues vocoder to convert mel spectrogram to audio.

        Args:
            x: (b, c, t, f)

        Outputs:
            y: (b, c, l)
        """
        return self.vocoder.decode(x)