from dataclasses import dataclass

from torch import nn

from splifft.models import ChunkSize, ModelConfigLike, ModelOutputStemName


@dataclass
class MyModelConfig(ModelConfigLike):  # (1)!
    chunk_size: ChunkSize
    output_stem_names: tuple[ModelOutputStemName, ...]
    # ... any other parameters your model needs


class MyModel(nn.Module):
    def __init__(self, cfg: MyModelConfig):
        super().__init__()
        self.cfg = cfg
