# ruff: noqa: E402
# --8<-- [start:config]
from pathlib import Path

from splifft.config import Config

config = Config.from_file(Path("path/to/my_model_config.json"))
# --8<-- [end:config]
# --8<-- [start:model]
from my_library.models import my_model_metadata

metadata = my_model_metadata()
my_model_config = config.model.to_concrete(metadata.config)
model = metadata.model(my_model_config)
# --8<-- [end:model]
# --8<-- [start:inference]
from splifft.inference import run_inference_on_file
from splifft.io import load_weights, read_audio

checkpoint_path = Path("path/to/my_model.pt")
model = load_weights(model, checkpoint_path, device="cpu")

mixture = read_audio(
    Path("path/to/mixture.wav"), config.audio_io.target_sample_rate, config.audio_io.force_channels
)
stems = run_inference_on_file(mixture, config, model)

print(f"{list(stems.keys())=}")
# --8<-- [end:inference]
