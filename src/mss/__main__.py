import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.logging import RichHandler

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="A CLI for source separation.",
    no_args_is_help=True,
)

# TODO: migrate away from hardcoding.
_DEFAULT_MODULE_NAME = "mss.models.bs_roformer"
_DEFAULT_CLASS_NAME = "BSRoformer"


@app.command()
def separate(
    mixture_path: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the audio file to be separated.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the model's JSON configuration file.",
        ),
    ],
    checkpoint_path: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the model's `.pt` or `.ckpt` checkpoint file.",
        ),
    ],
    module_name: Annotated[
        str,
        typer.Option(
            "--module",
            help=(
                "Python module containing the model and configuration class."
                f" Defaults to `{_DEFAULT_MODULE_NAME}` if not specified."
            ),
        ),
    ] = _DEFAULT_MODULE_NAME,
    class_name: Annotated[
        str,
        typer.Option(
            "--class",
            help=(
                "Name of the model class to load from the module."
                f" Defaults to `{_DEFAULT_CLASS_NAME}` if not specified."
            ),
        ),
    ] = _DEFAULT_CLASS_NAME,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Directory to save the separated audio stems.",
        ),
    ] = None,
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Force processing on CPU, even if CUDA is available.")
    ] = False,
) -> None:
    """Separates an audio file into its constituent stems."""
    import torch
    import torchaudio

    from .config import Config
    from .core import get_dtype
    from .inference import run_inference_on_file
    from .io import load_weights, read_audio
    from .models import ModelMetadata

    use_cuda = not cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"using {device=}")

    logger.info(f"loading configuration from {config_path=}")
    config = Config.from_file(config_path)

    logger.info(f"loading model metadata `{class_name}` from module `{module_name}`")
    model_metadata = ModelMetadata.from_module(
        module_name=module_name,
        model_cls_name=class_name,
        model_type=config.model_type,
    )
    config_model_concrete = config.model.to_concrete(model_metadata.config)
    model = model_metadata.model(config_model_concrete)
    logger.info(f"loading weights from {checkpoint_path=}")
    model = load_weights(model, checkpoint_path, device)
    model.to(get_dtype(config.inference.compute_dtype)).eval()

    logger.info(f"processing audio file: {mixture_path=}")
    mixture = read_audio(
        mixture_path,
        config.audio_io.target_sample_rate,
        config.audio_io.force_channels,
        device=device,
    )
    output_stems = run_inference_on_file(
        mixture,
        config=config,
        model=model,
    )
    if not config.output:
        return
    output_dir = output_dir or Path("./data/audio/output") / mixture_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem_name, stem_data in output_stems.items():
        if config.output.stem_names != "all" and stem_name not in config.output.stem_names:
            logger.info(
                f"skipping stem `{stem_name}` as it is not in the configured output stems: {config.output.stem_names=}"
            )
            continue

        output_file = (output_dir / stem_name).with_suffix(f".{config.output.file_format}")
        torchaudio.save(
            output_file,
            stem_data.cpu(),
            mixture.sample_rate,
            format=config.output.file_format,
            encoding=config.output.audio_encoding,
            bits_per_sample=config.output.bit_depth,
            # TODO: compression, backend
        )
        logger.info(f"wrote stem `{stem_name}` to {output_file}")


if __name__ == "__main__":
    app()
