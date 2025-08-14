from dataclasses import dataclass

from torch import nn

from splifft.models import ChunkSize, ModelIoType, ModelOutputStemName, ModelParamsLike


@dataclass
class MyModelParams(ModelParamsLike):  # (1)!
    chunk_size: ChunkSize
    output_stem_names: tuple[ModelOutputStemName, ...]
    # ... any other config your model needs
    input_type: ModelIoType = "waveform"
    output_type: ModelIoType = "waveform"


class MyModel(nn.Module):
    def __init__(self, params: MyModelParams):
        super().__init__()
        self.params = params
