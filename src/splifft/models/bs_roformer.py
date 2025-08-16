from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint

from . import (
    ChunkSize,
    Dropout,
    Ge0,
    Gt0,
    ModelInputType,
    ModelOutputStemName,
    ModelOutputType,
    ModelParamsLike,
)
from .utils.attend import Attend

try:
    from .utils.attend_sage import AttendSage
except ImportError:
    pass
if TYPE_CHECKING:
    from ..config import TorchDtype

# fmt: off
DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129
)
# fmt: on


@dataclass
class BSRoformerParams(ModelParamsLike):
    chunk_size: ChunkSize
    output_stem_names: tuple[ModelOutputStemName, ...]
    dim: Gt0[int]
    depth: Gt0[int]
    stereo: bool = True
    time_transformer_depth: Gt0[int] = 1
    freq_transformer_depth: Gt0[int] = 1
    linear_transformer_depth: Ge0[int] = 0
    freqs_per_bands: tuple[Gt0[int], ...] = field(default_factory=lambda: DEFAULT_FREQS_PER_BANDS)
    dim_head: int = 64
    heads: Gt0[int] = 8
    attn_dropout: Dropout = 0.0
    ff_dropout: Dropout = 0.0
    ff_mult: Gt0[int] = 4
    flash_attn: bool = True
    mask_estimator_depth: Gt0[int] = 2
    mlp_expansion_factor: Gt0[int] = 4
    use_torch_checkpoint: bool = False
    sage_attention: bool = False
    use_shared_bias: bool = False  # COMPAT: weights are all zeros anyways, disabling by default
    skip_connection: bool = False  # NOTE: not yet implemented
    rms_norm_eps: Ge0[float] | None = None
    rotary_embed_dtype: TorchDtype | None = None
    transformer_residual_dtype: TorchDtype | None = None
    debug: bool = False
    """Whether to check for nan/inf in model outputs. Keep it off for [torch.compile][]."""

    input_type: ModelInputType = "spectrogram"
    output_type: ModelOutputType = "spectrogram_mask"


def l2norm(t: Tensor) -> Tensor:
    return F.normalize(t, dim=-1, p=2)


class RMSNorm(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma  # type: ignore


class RMSNormWithEps(Module):
    def __init__(self, dim: int, eps: float = 5.960464477539063e-08):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        l2_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        denom = torch.maximum(l2_norm, torch.full_like(l2_norm, self.eps))
        normalized_x = x / denom
        return normalized_x * self.scale * self.gamma  # type: ignore


def rms_norm(dim: int, eps: float | None) -> RMSNorm | RMSNormWithEps:
    if eps is None:
        return RMSNorm(dim)
    return RMSNormWithEps(dim, eps)


# attention


class RotaryEmbedding(nn.Module):
    """A performance-oriented version of RoPE.

    Unlike `lucidrains`' implementation which compute embeddings JIT during the
    forward pass and caches calls with the same or shorter sequence length,
    we simply compute them AOT as persistent buffers. To keep the computational
    graph clean, we do not support dynamic sequence lengths, learned frequencies
    or length extrapolation.
    """

    def __init__(
        self, seq_len: int, dim_head: int, *, dtype: torch.dtype | None, theta: int = 10000
    ):
        super().__init__()
        # COMPAT: the original implementation does not generate the embeddings
        # on the fly, but serialises them in fp16. there are some tiny
        # differences:
        # |                     |   from weights  |   generated    |
        # | ------------------- | --------------- | -------------- |
        # | cos_emb_time:971,22 | -0.99462890625  | -0.994140625   |
        # | cos_emb_time:971,23 | -0.99462890625  | -0.994140625   |
        # | sin_emb_time:727,12 | -0.457763671875 | -0.4580078125  |
        # | sin_emb_time:727,13 | -0.457763671875 | -0.4580078125  |
        # | sin_emb_time:825,4  | -0.8544921875   | -0.85400390625 |
        # | sin_emb_time:825,5  | -0.8544921875   | -0.85400390625 |
        freqs = 1.0 / (theta ** (torch.arange(0, dim_head, 2).float() / dim_head))
        t = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", t, freqs)  # (seq_len, dim / 2)
        freqs = repeat(freqs, "... d -> ... (d r)", r=2)  # (seq_len, dim)
        self.cos_emb = freqs.cos().to(dtype)
        self.sin_emb = freqs.sin().to(dtype)

    def rotate_half(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    def forward(self, x: Tensor) -> Tensor:
        # x is (batch_eff, heads, seq_len_for_rotation, dim_head)
        cos_b = self.cos_emb.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        sin_b = self.sin_emb.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)

        term1 = x * cos_b
        term2 = self.rotate_half(x) * sin_b

        # NOTE: original impl performed addition between two f32s but it comes with 30% slowdown
        # we eliminate it so the addition is performed between two f16s (according to __init__).
        return term1 + term2


class FeedForward(Module):
    def __init__(
        self, dim: int, mult: int = 4, dropout: float = 0.0, rms_norm_eps: float | None = None
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            rms_norm(dim, eps=rms_norm_eps),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        shared_qkv_bias: nn.Parameter | None = None,
        shared_out_bias: nn.Parameter | None = None,
        rotary_embed: RotaryEmbedding | None = None,
        flash: bool = True,
        sage_attention: bool = False,
        rms_norm_eps: float | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        if sage_attention:
            self.attend = AttendSage(flash=flash, dropout=dropout)
        else:
            self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = rms_norm(dim, eps=rms_norm_eps)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=(shared_out_bias is not None)),
            nn.Dropout(dropout),
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)

        if self.rotary_embed is not None:
            q = self.rotary_embed(q)
            k = self.rotary_embed(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        gate_act = gates.sigmoid()

        out = out * rearrange(gate_act, "b n h -> b h n 1")

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 32,
        heads: int = 8,
        scale: int = 8,
        flash: bool = False,
        dropout: float = 0.0,
        sage_attention: bool = False,
        rms_norm_eps: float | None = None,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = rms_norm(dim, eps=rms_norm_eps)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        if sage_attention:
            self.attend = AttendSage(scale=scale, dropout=dropout, flash=flash)  # type: ignore
        else:
            self.attend = Attend(scale=scale, dropout=dropout, flash=flash)  # type: ignore

        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"), nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        norm_output: bool = True,
        rotary_embed: RotaryEmbedding | None = None,
        flash_attn: bool = True,
        linear_attn: bool = False,
        sage_attention: bool = False,
        shared_qkv_bias: nn.Parameter | None = None,
        shared_out_bias: nn.Parameter | None = None,
        rms_norm_eps: float | None = None,
        transformer_residual_dtype: torch.dtype | None = None,  # COMPAT: float32, see 265
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn: LinearAttention | Attention
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                    rms_norm_eps=rms_norm_eps,
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    shared_qkv_bias=shared_qkv_bias,
                    shared_out_bias=shared_out_bias,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                    rms_norm_eps=rms_norm_eps,
                )

            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, rms_norm_eps=rms_norm_eps)
            self.layers.append(ModuleList([attn, ff]))
        self.transformer_residual_dtype = transformer_residual_dtype

        self.norm = rms_norm(dim, eps=rms_norm_eps) if norm_output else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:  # type: ignore
            attn_out = attn(x)
            if self.transformer_residual_dtype is not None:
                x = (
                    attn_out.to(self.transformer_residual_dtype)
                    + x.to(self.transformer_residual_dtype)
                ).to(x.dtype)
            else:
                x = attn_out + x

            ff_out = ff(x)
            if self.transformer_residual_dtype is not None:
                x = (
                    ff_out.to(self.transformer_residual_dtype)
                    + x.to(self.transformer_residual_dtype)
                ).to(x.dtype)
            else:
                x = ff_out + x
        return self.norm(x)


# bandsplit module


class BandSplit(Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], rms_norm_eps: float | None = None):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(rms_norm(dim_in, rms_norm_eps), nn.Linear(dim_in, dim))
            self.to_features.append(net)

    def forward(self, x: Tensor) -> Tensor:
        x_split = x.split(self.dim_inputs, dim=-1)
        outs = []
        for split_input, to_feature_net in zip(x_split, self.to_features):
            split_output = to_feature_net(split_input)
            outs.append(split_output)
        return torch.stack(outs, dim=-2)


def mlp(
    dim_in: int,
    dim_out: int,
    dim_hidden: int | None = None,
    depth: int = 1,
    activation: type[Module] = nn.Tanh,
) -> nn.Sequential:
    dim_hidden_ = dim_hidden or dim_in

    net: list[Module] = []
    dims = (dim_in, *((dim_hidden_,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(
        self, dim: int, dim_inputs: tuple[int, ...], depth: int, mlp_expansion_factor: int
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            self.to_freqs.append(
                nn.Sequential(
                    mlp(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1)
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        x_unbound = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x_unbound, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


class BSRoformer(Module):
    def __init__(self, cfg: BSRoformerParams):
        super().__init__()
        self.stereo = cfg.stereo
        self.audio_channels = 2 if cfg.stereo else 1
        self.num_stems = len(cfg.output_stem_names)
        self.use_torch_checkpoint = cfg.use_torch_checkpoint
        self.skip_connection = cfg.skip_connection

        self.layers = ModuleList([])

        self.shared_qkv_bias: nn.Parameter | None = None
        self.shared_out_bias: nn.Parameter | None = None
        if cfg.use_shared_bias:
            dim_inner = cfg.heads * cfg.dim_head
            self.shared_qkv_bias = nn.Parameter(torch.ones(dim_inner * 3))
            self.shared_out_bias = nn.Parameter(torch.ones(cfg.dim))

        transformer = partial(
            Transformer,
            dim=cfg.dim,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout,
            ff_mult=cfg.ff_mult,
            flash_attn=cfg.flash_attn,
            norm_output=False,
            sage_attention=cfg.sage_attention,
            shared_qkv_bias=self.shared_qkv_bias,
            shared_out_bias=self.shared_out_bias,
            rms_norm_eps=cfg.rms_norm_eps,
            transformer_residual_dtype=cfg.transformer_residual_dtype,
        )

        # NOTE: hardcoding stft hop length for now - we should parameterise this.
        stft_hop_length = 512
        t_frames = cfg.chunk_size // stft_hop_length + 1  # e.g. 588800 // 512 + 1 = 1151
        time_rotary_embed = RotaryEmbedding(
            seq_len=t_frames, dim_head=cfg.dim_head, dtype=cfg.rotary_embed_dtype
        )

        num_bands = len(cfg.freqs_per_bands)
        freq_rotary_embed = RotaryEmbedding(
            seq_len=num_bands, dim_head=cfg.dim_head, dtype=cfg.rotary_embed_dtype
        )

        for _ in range(cfg.depth):
            tran_modules = []
            if cfg.linear_transformer_depth > 0:
                tran_modules.append(
                    transformer(depth=cfg.linear_transformer_depth, linear_attn=True)
                )
            tran_modules.append(
                transformer(depth=cfg.time_transformer_depth, rotary_embed=time_rotary_embed)
            )
            tran_modules.append(
                transformer(depth=cfg.freq_transformer_depth, rotary_embed=freq_rotary_embed)
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = rms_norm(cfg.dim, eps=cfg.rms_norm_eps)

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in cfg.freqs_per_bands
        )

        self.band_split = BandSplit(
            dim=cfg.dim, dim_inputs=freqs_per_bands_with_complex, rms_norm_eps=cfg.rms_norm_eps
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(len(cfg.output_stem_names)):
            mask_estimator = MaskEstimator(
                dim=cfg.dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=cfg.mask_estimator_depth,
                mlp_expansion_factor=cfg.mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        self.debug = cfg.debug

    def forward(self, stft_repr: Tensor) -> Tensor:
        """
        :param stft_repr: input spectrogram. shape (b, f*s, t, c)
        :return: estimated mask. shape (b, n, f, t, c)
        """
        x = rearrange(stft_repr, "b f t c -> b t (f c)")

        if self.debug and (torch.isnan(x).any() or torch.isinf(x).any()):
            raise RuntimeError(
                f"nan/inf in x after rearrange: {x.isnan().sum()} nans, {x.isinf().sum()} infs"
            )

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        if self.debug and (torch.isnan(x).any() or torch.isinf(x).any()):
            raise RuntimeError(
                f"nan/inf in x after band_split: {x.isnan().sum()} nans, {x.isinf().sum()} infs"
            )

        # axial / hierarchical attention

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], "b * d")
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        if self.use_torch_checkpoint:
            mask = torch.stack(
                [checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators],
                dim=1,
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        return mask
