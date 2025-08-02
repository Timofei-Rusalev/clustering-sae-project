# sae_v5_32k.py
# ---------------------------------------------------------------------
# Sparse Auto‑Encoder v5‑32k
#
# * One SAE instance per GPT‑2 layer
# * 32 768 latent features (default)
# * Top‑K sparse activation + optional layer‑norm
#
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import torch

__all__ = ["SingleSAEv5", "load_sae_layer"]

EPS: float = 1e-5  # small constant for numerical stability


class SingleSAEv5(torch.nn.Module):
    """
    Sparse Auto‑Encoder (SAE v5‑32k) for a single transformer layer.

    Parameters are loaded from a checkpoint `layer*.pt` that must contain:
      • pre_bias              :  (d,)
      • latent_bias           :  (F,)
      • encoder.weight        :  (F, d)
      • decoder.weight        :  (d, F)
      • activation_state_dict :  {"k": <top‑K>}
    Where:
      d – hidden‑state dimension (e.g. 768 for GPT‑2 small)
      F – number of latent neurons (default 32 768)
    """

    def __init__(self, sd: Dict[str, Any], *, device: str = "cpu") -> None:
        super().__init__()

        # latent / hidden sizes
        self.d: int = sd["pre_bias"].numel()
        self.F: int = sd["latent_bias"].numel()

        # --- Bias buffers (do not require gradients)
        self.register_buffer("pre_bias", sd["pre_bias"].to(device))
        self.register_buffer("latent_bias", sd["latent_bias"].to(device))

        # --- Encoder / decoder weights
        self.encoder = torch.nn.Linear(self.d, self.F, bias=False, device=device)
        self.decoder = torch.nn.Linear(self.F, self.d, bias=False, device=device)
        with torch.no_grad():
            self.encoder.weight.copy_(sd["encoder.weight"].to(device))
            self.decoder.weight.copy_(sd["decoder.weight"].to(device))

        # --- Top‑K sparsification
        act_cfg = sd["activation_state_dict"]
        self.k: int = int(act_cfg["k"])
        self.normalize: bool = True  # layer‑norm before encoding

    # ------------------------------------------------------------------ #
    # Static helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _layer_norm(x: torch.Tensor, eps: float = EPS) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Manual layer‑norm: return normalised tensor + (mean, std) for de‑norm.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        std = (var + eps).sqrt()
        return (x - mean) / std, mean, std

    # ------------------------------------------------------------------ #
    # Forward helpers                                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode residual `x` → sparse latent `h`.

        Parameters
        ----------
        x : (..., d)  – residual hidden states

        Returns
        -------
        h    : (..., F)                – sparse activations (Top‑K ReLU)
        info : {"mean": μ, "std": σ}   – stats for (optional) de‑normalisation
        """
        if self.normalize:
            x_norm, mean, std = self._layer_norm(x)
        else:
            x_norm, mean, std = x, None, None

        # centre & project
        x_centered = x_norm - self.pre_bias
        h_pre = torch.nn.functional.linear(x_centered, self.encoder.weight, self.latent_bias)

        # Top‑K + ReLU
        topk_vals, topk_idx = torch.topk(h_pre, self.k, dim=-1)
        h = torch.zeros_like(h_pre)
        h.scatter_(-1, topk_idx, torch.nn.functional.relu(topk_vals))

        return h, {"mean": mean, "std": std}

    @torch.no_grad()
    def decode(self, h: torch.Tensor, info: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Reconstruct residual from sparse latent `h`.

        Parameters
        ----------
        h    : (..., F)                – sparse activations
        info : {"mean": μ, "std": σ}   – stats from `encode` (for de‑norm)

        Returns
        -------
        x_hat : (..., d)               – reconstructed residual
        """
        x_rec_norm = torch.nn.functional.linear(h, self.decoder.weight) + self.pre_bias
        if self.normalize and info is not None:
            return x_rec_norm * info["std"] + info["mean"]
        return x_rec_norm


# --------------------------------------------------------------------------
# Convenience loader
# --------------------------------------------------------------------------
def load_sae_layer(ckpt_path: Path | str, *, device: str = "cpu") -> SingleSAEv5:
    """
    Load a `layer*.pt` checkpoint and return an initialised `SingleSAEv5`.

    Example
    -------
    >>> sae = load_sae_layer("checkpoints/layer3.pt", device="cuda")
    """
    state_dict = torch.load(str(ckpt_path), map_location=device)
    return SingleSAEv5(state_dict, device=device)