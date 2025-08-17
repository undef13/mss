from dataclasses import dataclass

from torch import nn

from splifft.models import ModelParamsLike
from splifft.types import ChunkSize, ModelInputType, ModelOutputStemName, ModelOutputType


@dataclass
class MyModelParams(ModelParamsLike):  # (1)!
    chunk_size: ChunkSize
    output_stem_names: tuple[ModelOutputStemName, ...]
    # ... any other config your model needs
    @property
    def input_type(self) -> ModelInputType:
        return "waveform"
    @property
    def output_type(self) -> ModelOutputType:
        return "waveform"


class MyModel(nn.Module):
    def __init__(self, params: MyModelParams):
        super().__init__()
        self.params = params
