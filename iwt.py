from dataclasses import dataclass, field
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor, nn


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    device = pos.device
    dtype = torch.float32 if device.type == 'mps' else torch.float64
    scale = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


class EmbedND(nn.Module):

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3
        )

        return emb.unsqueeze(1)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim)

    __call__: Callable[[Tensor], tuple[ModulationOut, ModulationOut | None]]

    def forward(self, key: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(key))[:, None, :].chunk(self.multiplier, dim=-1)

        return (ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None)


def modulate(x: Tensor, mod: ModulationOut) -> Tensor:
    return (1 + mod.scale) * x + mod.shift


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim, eps=1e-6)
        self.key_norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe)
        x = self.proj(x)
        return x


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.x_mod = Modulation(hidden_size, double=True)
        self.x_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.x_attn = SelfAttention(hidden_size, num_heads, qkv_bias)

        self.x_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.x_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        self.wm_mod = Modulation(hidden_size, double=True)
        self.wm_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.wm_attn = SelfAttention(hidden_size, num_heads, qkv_bias)

        self.wm_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.wm_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, x: Tensor, wm: Tensor, key: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        x_mod1, x_mod2 = self.x_mod(key)
        wm_mod1, wm_mod2 = self.wm_mod(key)

        # prepare x for attention
        x_modulated = modulate(self.x_norm1(x), x_mod1)
        x_qkv = self.x_attn.qkv(x_modulated)
        x_q, x_k, x_v = rearrange(x_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        x_q, x_k = self.x_attn.norm(x_q, x_k, x_v)

        # prepare wm for attention
        wm_modulated = modulate(self.wm_norm1(wm), wm_mod1)
        wm_qkv = self.wm_attn.qkv(wm_modulated)
        wm_q, wm_k, wm_v = rearrange(wm_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        wm_q, wm_k = self.wm_attn.norm(wm_q, wm_k, wm_v)

        # run actual attention
        q = torch.cat((wm_q, x_q), dim=2)
        k = torch.cat((wm_k, x_k), dim=2)
        v = torch.cat((wm_v, x_v), dim=2)

        attn = attention(q, k, v, pe)
        wm_attn, s_attn = attn[:, : wm.shape[1]], attn[:, wm.shape[1] :]

        # calculate the x bloks
        x = x + x_mod1.gate * self.x_attn.proj(s_attn)
        x = x + x_mod2.gate * self.x_mlp(modulate(self.x_norm2(x), x_mod2))

        # calculate the wm bloks
        wm = wm + wm_mod1.gate * self.wm_attn.proj(wm_attn)
        wm = wm + wm_mod2.gate * self.wm_mlp(modulate(self.wm_norm2(wm), wm_mod2))
        return x, wm


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, key: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(key)
        x_mod = modulate(self.pre_norm(x), mod)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, key: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(key).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


scale = 3


@dataclass
class IWTParams:
    h: int = 128
    w: int = 128
    lw: int = 64
    lk: int = 32
    in_channels: int = 12
    hidden_size: int = 3072 // scale
    mlp_ratio: float = 4.0
    num_heads: int = 24 // scale
    encoder_depth_double_blocks: int = 1
    encoder_depth_single_blocks: int = 2
    decoder_depth: int = 3
    axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    theta: int = 10_000
    qkv_bias: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Hidden size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        self.pe_dim = self.hidden_size // self.num_heads
        if sum(self.axes_dim) != self.pe_dim:
            raise ValueError(f"Got {self.axes_dim} but expected positional dim {self.pe_dim}")


class Base(nn.Module):

    def __init__(self, params: IWTParams = IWTParams()):
        super().__init__()

        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.x_in = nn.Linear(self.in_channels, self.hidden_size)
        # TODO: key变1位，变化要大
        self.key_in = nn.Linear(params.lk, self.hidden_size)
        self.pe_embedder = EmbedND(params.pe_dim, params.theta, params.axes_dim)

    def get_ids(self, h: int, w: int, bs: int) -> Tensor:
        ids = torch.zeros(h, w, 3)
        ids[..., 1] = ids[..., 1] + torch.arange(h)[:, None]
        ids[..., 2] = ids[..., 2] + torch.arange(w)[None, :]
        ids = repeat(ids, 'h w c -> b (h w) c', b=bs)
        return ids


class Encoder(Base):

    def __init__(self, params: IWTParams = IWTParams()):
        super().__init__(params)

        # TODO: w [N, Lw] -> [N, Lw, hidden_size]
        self.wm_in = nn.Embedding(2, self.hidden_size)
        self.double_blocks = nn.ModuleList(
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                params.mlp_ratio,
                params.qkv_bias,
            )
            for _ in range(params.encoder_depth_double_blocks)
        )
        self.single_blocks = nn.ModuleList(
            SingleStreamBlock(self.hidden_size, self.num_heads, params.mlp_ratio)
            for _ in range(params.encoder_depth_single_blocks)
        )
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(self, x: Tensor, wm: Tensor, key: Tensor) -> Tensor:
        bs, c, h, w = x.shape

        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        x = self.x_in(x)
        wm = self.wm_in(wm)
        key = self.key_in(key)

        x_ids = self.get_ids(h // 2, w // 2, bs)
        wm_ids = self.get_ids(int(wm.shape[1] ** 0.5), int(wm.shape[1] ** 0.5), bs)
        ids = torch.cat((wm_ids, x_ids), dim=1).to(x.device)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            x, wm = block(x, wm, key, pe)

        x = torch.cat((wm, x), 1)
        for block in self.single_blocks:
            x = block(x, key, pe)
        x = x[:, wm.shape[1] :, ...]

        x = self.final_layer(x, key)  # (N, T, patch_size ** 2 * out_channels)
        x = rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h // 2, w=w // 2, ph=2, pw=2)
        return x


class Decoder(Base):

    def __init__(self, params: IWTParams = IWTParams()):
        super().__init__(params)

        self.single_blocks = nn.ModuleList(
            SingleStreamBlock(self.hidden_size, self.num_heads, params.mlp_ratio)
            for _ in range(params.decoder_depth)
        )
        # TODO: x to w，分辨率无关，任意数变lw
        self.linear1 = nn.Linear(self.hidden_size, 1)
        self.linear2 = nn.Linear(params.h * params.w // 4, params.lw)

    def forward(self, x: Tensor, key: Tensor) -> Tensor:
        bs, c, h, w = x.shape

        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        x = self.x_in(x)
        key = self.key_in(key)

        ids = self.get_ids(h // 2, w // 2, bs).to(x.device)
        pe = self.pe_embedder(ids)

        for block in self.single_blocks:
            x = block(x, key, pe)

        wm = self.linear2(self.linear1(x).squeeze(2))
        return wm


class IWT(nn.Module):

    def __init__(self, params: IWTParams = IWTParams()):
        super().__init__()

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x: Tensor, wm: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        x_encoded = self.encoder(x, wm, key)
        wm_decoded = self.decoder(x_encoded, key)
        return x_encoded, wm_decoded

    def encode(self, x: Tensor, wm: Tensor, key: Tensor) -> Tensor:
        return self.encoder(x, wm, key)

    def encode(self, x: Tensor, key: Tensor) -> Tensor:
        return self.decoder(x, key)


def main():
    bs = 2
    params = IWTParams()

    iwt = IWT(params)

    x = torch.rand(bs, 3, params.h, params.w) * 2 - 1
    wm = torch.randint(0, 2, [bs, params.lw])
    key = torch.randint(0, 2, [bs, params.lk]) - 0.5

    device = torch.device('mps')
    iwt = iwt.to(device)
    x = x.to(device)
    wm = wm.to(device)
    key = key.to(device)

    x_encoded, wm_decoded = iwt(x, wm, key)

    ...


if __name__ == '__main__':
    main()
