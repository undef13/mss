from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import log_once, parse_version

logger = logging.getLogger(__name__)

try:
    from sageattention import sageattn  # type: ignore

    _has_sage_attention = True
except ImportError:
    _has_sage_attention = False
    log_once(
        logger,
        "sageattention not found. Will fall back to PyTorch SDPA (if available) or manual einsum.",
    )


class AttendSage(nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        flash: bool = False,
        scale: float | None = None,
    ):
        """
        :param flash: if True, attempts to use SageAttention or PyTorch SDPA.
        """
        super().__init__()
        self.scale = scale  # for einsum path
        self.dropout = dropout  # for einsum/SDPA path

        self.use_sage = flash and _has_sage_attention
        self.use_pytorch_sdpa = False
        self._sdpa_checked = False

        if flash and not self.use_sage:
            if not self._sdpa_checked:
                if parse_version(torch.__version__) >= (2, 0, 0):
                    self.use_pytorch_sdpa = True
                    log_once(
                        logger,
                        "Using PyTorch SDPA backend (FlashAttention-2, Memory-Efficient, or Math).",
                    )
                else:
                    log_once(
                        logger,
                        "Flash attention requested but Pytorch < 2.0 and SageAttention not found. Falling back to einsum.",
                    )
                self._sdpa_checked = True

        # dropout layer for manual einsum implementation ONLY
        # SDPA and SageAttention handle dropout differently
        # (or not at all in Sage's base API)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        einstein notation

        - b: batch
        - h: heads
        - n, i, j: sequence length (base sequence length, source, target)
        - d: feature dimension

        Input tensors q, k, v expected in shape: (batch, heads, seq_len, dim_head) -> HND layout
        """
        _q_len, _k_len, _device = q.shape[-2], k.shape[-2], q.device

        # priority 1: SageAttention
        if self.use_sage:
            # assumes q, k, v are FP16/BF16 (handled by autocast upstream)
            # assumes scale is handled internally by sageattn
            # assumes dropout is NOT handled by sageattn kernel
            # is_causal=False based on how Attend is called in mel_band_roformer
            out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)  # type: ignore
            return out  # type: ignore
            try:
                out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
                return out
            except Exception as e:
                logger.error(f"SageAttention failed with error: {e}. Falling back.")
                self.use_sage = False
                if not self._sdpa_checked:
                    if parse_version(torch.__version__) >= (2, 0, 0):
                        self.use_pytorch_sdpa = True
                        log_once(logger, "falling back to PyTorch SDPA")
                    else:
                        log_once(logger, "falling back to einsum.")

                    self._sdpa_checked = True

        # priority 2: PyTorch SDPA
        if self.use_pytorch_sdpa:
            # it handles scaling and dropout internally.
            try:
                with sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
                ):
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,  # assuming no explicit mask needed here
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=False,  # assuming not needed based on usage context
                    )
                return out
            except Exception as e:
                log_once(
                    logger,
                    f"pytorch SDPA failed with error: {e}. falling back to einsum.",
                    level=logging.ERROR,
                )
                self.use_pytorch_sdpa = False

        scale = self.scale or q.shape[-1] ** -0.5

        # similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)  # ONLY in einsum path

        # aggregate values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
