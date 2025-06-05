# mss

Lightweight utilities for music source separation.

## Installation

```sh
uv add git+https://github.com/undef13/mss.git
```

## Development

```sh
uv venv
uv sync --all-extras --all-groups
# serve documentation.
uv mkdocs serve
```

## Usage

Note that this repo is no longer compatible with zfturbo's repo. The last version that does so is `v0.0.1`. To pin the version in uv, change your `pyproject.toml`:

```toml
[tool.uv.sources]
mss = { git = "https://github.com/undef13/mss.git", rev = "287235e520f3bb927b58f9f53749fe3ccc248fac" }
```

### Documentation for v0.0.1

To reproduce, modify zfturbo's source:

```patch
diff --git a/utils/model_utils.py b/utils/model_utils.py
--- a/utils/model_utils.py
+++ b/utils/model_utils.py
@@ -51,7 +51,7 @@ def demix(
         - A numpy array of the separated source if only one instrument is present.
     """
 
-    mix = torch.tensor(mix, dtype=torch.float32)
+    mix = torch.tensor(mix, dtype=torch.float16)
 
     if model_type == 'htdemucs':
         mode = 'demucs'
diff --git a/utils/settings.py b/utils/settings.py
--- a/utils/settings.py
+++ b/utils/settings.py
@@ -260,9 +260,9 @@ def get_model_from_config(model_type: str, config_path: str) -> Tuple:
         model = BSRoformer(**dict(config.model))
+    elif model_type == 'bs_roformer_dev':
+        from mss.models.bs_roformer import BSRoformer
+        model = BSRoformer(**dict(config.model)).half()
     elif model_type == 'bs_roformer_experimental':
         from models.bs_roformer.bs_roformer_experimental import BSRoformer
         model = BSRoformer(**dict(config.model))
```


```yml
audio:
  chunk_size: 588800
  dim_f: 1024
  dim_t: 801
  hop_length: 441
  n_fft: 2048
  num_channels: 2
  sample_rate: 44100
  min_mean_abs: 0.000

model:
  dim: 256
  depth: 12
  stereo: true
  num_stems: 6
  time_transformer_depth: 1
  freq_transformer_depth: 1
  use_shared_bias: True

training:
  instruments: ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano']
  use_amp: true # enable fp16 support

inference:
  batch_size: 8
  dim_t: 1101
  num_overlap: 2
  normalize: false
```

```sh
cd ../music-source-separation-training && uv run inference.py \
    --model_type bs_roformer_dev \
    --config_path ./path/to/roformer.yaml \
    --start_check_point ./path/to/roformer.pt \
    --input_folder ./path/to/input \
    --store_dir ./path/to/output
```

