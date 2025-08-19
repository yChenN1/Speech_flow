import torch
import torch.nn as nn
from einops import rearrange
from torch import LongTensor, Tensor

def generate_coords_sequence(H, W, device=None):
    """
    生成形状为 (H*W, 2) 的坐标序列，长度为 H*W，每行对应一个 (y, x) 坐标
    
    Args:
        H: 高度（行数）
        W: 宽度（列数）
        device: 设备（如 torch.device('cuda')）
    
    Returns:
        coords_seq: 形状为 (H*W, 2) 的张量，序列顺序为「先行后列」
    """
    # 生成行索引 (H,) 和列索引 (W,)
    y = torch.arange(H, device=device)
    x = torch.arange(W, device=device)
    
    # 创建网格，得到 (H, W) 形状的 y 坐标和 x 坐标（先行后列）
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    # 展平为 (H*W,) 的一维数组，再拼接为 (H*W, 2) 的序列
    y_flat = y_grid.flatten()  # 按先行后列展平行坐标
    x_flat = x_grid.flatten()  # 按先行后列展平列坐标
    coords_seq = torch.stack([y_flat, x_flat], dim=1)  # 拼接为序列
    
    return coords_seq

class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 8192, base: int = 10000, device=None):
        r"""Rotary position embedding.

        [1] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary 
        position embedding." Neurocomputing, 2024

        h: head_dim
        l: seq_len
        """
        super().__init__()

        self.head_dim = head_dim

        # Calculate θ = 1 / 10000**(2i/h)
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))  # (h/2,)

        # Matrix pθ
        pos_theta = torch.outer(torch.arange(max_len), theta).float()  # (l, h/2)

        # Rotation matrix
        w = torch.stack([torch.cos(pos_theta), torch.sin(pos_theta)], dim=-1)  # (l, h/2, 2)
        self.w = w.to(device)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply RoPE.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, l, n, h)

        Outputs:
            out: (b, l, n, h)
        """
        L = x.shape[1]
        x = rearrange(x, 'b l n (h c) -> b l n h c', c=2)  # (b, l, n, h/2, 2)
        w = self.w[0 : L][None, :, None, :, :]  # (1, l, 1, h/2, 2)
        x = self.rotate(x, w)  # (b, l, n, h/2, 2)
        x = rearrange(x, 'b l n h c -> b l n (h c)')  # (b, l, n, h)
        
        return x

    def rotate(self, x: Tensor, w: Tensor) -> Tensor:
        r"""Rotate x.

        x0 = cos(θp)·x0 - sin(θp)·x1
        x1 = sin(θp)·x0 + cos(θp)·x1

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim

        Args:
            x: (b, l, n, h/2, 2)
            w: (1, l, 1, h/2, 2)

        Outputs:
            out: (b, l, n, h/2, 2)
        """

        out = torch.stack([
            w[..., 0] * x[..., 0] - w[..., 1] * x[..., 1],
            w[..., 0] * x[..., 1] + w[..., 1] * x[..., 0]
            ],
            dim=-1,
        )  # (b, l, n, h/2, 2)

        return out

    def apply_nd_sparse(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply Nd RoPE with sparse positions.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim
        k: data dim

        Args:
            x: (b, l, n, h)
            pos: (l, k)
            n_dim: int

        Outputs:
            out: (b, l, n, h)
        """
        
        B, L, N, H = x.shape
        K = pos.shape[1]  # rope_dim
        assert H == K * self.head_dim

        x = rearrange(x, 'b l n (k h c) -> k b l n h c', k=K, c=2)  # (k, b, l, n, h/2, 2)
        x = x.contiguous()

        for i in range(K):
            p = pos[:, i]  # (l,)
            w = self.w[p][None, :, None, :, :]  # (1, l, 1, h/2, 2)
            x[i] = self.rotate(x[i], w).clone()  # x: (k, b, l, n, h/2, 2)

        out = rearrange(x, 'k b l n h c -> b l n (k h c)')  # (b, l, n, h)
        
        return out
    
    def apply_nd(self, x: Tensor, grid_size) -> Tensor:
        r"""Apply Nd RoPE with continous positions.

        b: batch_size
        l: seq_len
        n: heads_num
        h: head_dim
        k: data dim

        Args:
            x: (b, l, n, h)
            pos: (l, k)
            n_dim: int

        Outputs:
            out: (b, l, n, h)
        """
        

        pos = generate_coords_sequence(*grid_size, device=x.device)
        B, L, N, H = x.shape
        K = pos.shape[1]  # rope_dim
        assert H == K * self.head_dim

        x = rearrange(x, 'b l n (k h c) -> k b l n h c', k=K, c=2)  # (k, b, l, n, h/2, 2)
        x = x.contiguous()

        for i in range(K):
            p = pos[:, i]  # (l,)
            w = self.w[p][None, :, None, :, :]  # (1, l, 1, h/2, 2)
            x[i] = self.rotate(x[i], w).clone()  # x: (k, b, l, n, h/2, 2)

        out = rearrange(x, 'k b l n h c -> b l n (k h c)')  # (b, l, n, h)
        
        return out