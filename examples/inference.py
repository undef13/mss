# ruff: noqa: E402
from pathlib import Path

PATH_CONFIG = Path("data/config/bs_roformer.json")
PATH_CKPT = Path("data/models/roformer-fp16.pt")
PATH_MIXTURE = Path("data/audio/input/3BFTio5296w.flac")

# 1. parse + validate a JSON *without* having to import a particular pytorch model.
from splifft.config import Config

config = Config.from_file(PATH_CONFIG)

# 2. we now want to *lock in* the configuration to a specific model.
from splifft.models import ModelMetadata
from splifft.models.bs_roformer import BSRoformer, BSRoformerParams

metadata = ModelMetadata(model_type="bs_roformer", params=BSRoformerParams, model=BSRoformer)
model_params = config.model.to_concrete(metadata.params)

# 3. `metadata` acts as a model builder
from splifft.io import load_weights

model = metadata.model(model_params)
model = load_weights(model, PATH_CKPT, device="cpu")

# 4. load audio and run inference by passing dependencies explicitly.
from splifft.inference import run_inference_on_file
from splifft.io import read_audio

mixture = read_audio(
    PATH_MIXTURE,
    config.audio_io.target_sample_rate,
    config.audio_io.force_channels,
)
stems = run_inference_on_file(mixture, config, model, model_params)

print(list(stems.keys()))
