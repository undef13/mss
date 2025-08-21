"""A high level API over the lower level functions for convenience.

!!! warning
    This module is experimental.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from splifft.config import Config
from splifft.core import Audio
from splifft.inference import run_inference_on_file
from splifft.io import load_weights, read_audio
from splifft.models import ModelMetadata

if TYPE_CHECKING:
    from splifft import types as t
    from splifft.config import StemName
    from splifft.models import ModelParamsLike


@dataclass
class SplifftModel:
    model: nn.Module
    config: Config
    model_params: ModelParamsLike

    @classmethod
    def from_pretrained(
        cls,
        config_path: t.BytesPath,
        checkpoint_path: t.StrPath,
        *,
        device: torch.device | str | None = None,
        module_name: str = "splifft.models.bs_roformer",
        class_name: str = "BSRoformer",
    ) -> SplifftModel:
        """Instantiates a model from a configuration and checkpoint file."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = Config.from_file(config_path)
        # a more robust impl would be to use a registry to map `config.model_type`
        # to the correct module and class, but it's not currently implemented.
        metadata = ModelMetadata.from_module(
            module_name,
            class_name,
            model_type=config.model_type,
        )
        model_params = config.model.to_concrete(metadata.params)
        model = metadata.model(model_params)
        if config.inference.force_weights_dtype:
            model = model.to(config.inference.force_weights_dtype)
        model = load_weights(model, checkpoint_path, device=device).eval()
        return cls(model=model, config=config, model_params=model_params)

    def separate(
        self,
        mixture: t.StrPath | t.BytesPath | t.RawAudioTensor | Audio[t.RawAudioTensor],
    ) -> dict[StemName, t.RawAudioTensor]:
        """
        Separates an audio file or tensor into its constituent stems.

        :param mixture: Path to the audio file, a raw audio tensor, or an `Audio` object.
        :return: A dictionary mapping stem names to their audio tensors.
        """
        if isinstance(mixture, Audio):
            audio_obj = mixture
        elif isinstance(mixture, torch.Tensor):
            audio_obj = Audio(data=mixture, sample_rate=self.config.audio_io.target_sample_rate)
        else:
            device = next(self.model.parameters()).device
            audio_obj = read_audio(
                mixture,  # type: ignore
                self.config.audio_io.target_sample_rate,
                self.config.audio_io.force_channels,
                device=device,
            )

        stems = run_inference_on_file(
            mixture=audio_obj,
            config=self.config,
            model=self.model,
            model_params_concrete=self.model_params,
        )
        return stems
