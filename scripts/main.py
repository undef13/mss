# %%
from functools import lru_cache
from pathlib import Path

import torch

from mss import PATH_DATA
from mss.models.bs_roformer import BSRoformer

model = BSRoformer(dim=128, depth=12, stereo=True).eval()
# %%

names_torch: dict[str, tuple[str, torch.nn.Parameter]] = {}
with torch.no_grad():
    for name, parameter in model.named_parameters():
        mil_name = f"{name.replace('.', '_')}_to_fp16"
        parameter.copy_(torch.full_like(parameter, torch.nan))
        names_torch[mil_name] = (name, parameter)
# %%
mil_to_torch_dtype = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


@lru_cache(maxsize=128)
def load_weights(path: Path) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    return data


def is_normal(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all() and not torch.isnan(tensor).any())


for name, parameter in model.named_parameters():
    # assert is_normal(parameter), f"{name=} {parameter=} {parameter.shape=}"
    print(name, parameter[:10])

model_weights = model.state_dict()
torch.save(model_weights, PATH_DATA / "weights.pt")

# %%
