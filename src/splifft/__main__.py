"""Command line interface for `splifft`."""

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
_DEFAULT_MODULE_NAME = "splifft.models.bs_roformer"
_DEFAULT_CLASS_NAME = "BSRoformer"


@app.command()
def separate(
    mixture_path: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=True,
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
            help="Python module containing the model and configuration class.",
        ),
    ] = _DEFAULT_MODULE_NAME,
    class_name: Annotated[
        str,
        typer.Option(
            "--class",
            help="Name of the model class to load from the module.",
        ),
    ] = _DEFAULT_CLASS_NAME,
    package_name: Annotated[
        Optional[str],
        typer.Option(
            "--package",
            help=(
                "The package to use as the anchor point from which to resolve the relative import "
                "to an absolute import. This is only required when performing a relative import."
            ),
        ),
    ] = None,
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
        package=package_name,
    )
    config_model_concrete = config.model.to_concrete(model_metadata.config)
    model = model_metadata.model(config_model_concrete)
    if config.inference.force_weights_dtype:
        model = model.to(get_dtype(config.inference.force_weights_dtype))
    logger.info(f"loading weights from {checkpoint_path=}")
    model = load_weights(model, checkpoint_path, device).eval()
    if (c := config.inference.compile_model) is not None:
        logger.info("enabled torch compilation")
        model = torch.compile(model, fullgraph=c.fullgraph, dynamic=c.dynamic, mode=c.mode)  # type: ignore

    mixture_paths = mixture_path.glob("*") if mixture_path.is_dir() else [mixture_path]
    for mixture_path in mixture_paths:
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
        curr_output_dir = output_dir or Path("./data/audio/output") / mixture_path.stem
        curr_output_dir.mkdir(parents=True, exist_ok=True)
        for stem_name, stem_data in output_stems.items():
            if config.output.stem_names != "all" and stem_name not in config.output.stem_names:
                continue

            output_file = (curr_output_dir / stem_name).with_suffix(f".{config.output.file_format}")
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


@app.command()
def debug() -> None:
    """Prints detailed information about the environment, dependencies, and hardware
    for debugging purposes."""
    import sys

    logger.info(f"{sys.version=}")
    logger.info(f"{sys.executable=}")
    logger.info(f"{sys.platform=}")
    import platform

    logger.info(f"{platform.system()=} ({platform.release()})")
    logger.info(f"{platform.machine()=}")
    import torch

    logger.info(f"{torch.__version__=}")
    logger.info(f"{torch.cuda.is_available()=}")
    if torch.cuda.is_available():
        logger.info(f"{torch.cuda.device_count()=}")
        logger.info(f"{torch.cuda.current_device()=}")
        device = torch.cuda.current_device()
        logger.info(f"{torch.cuda.get_device_name(device)=}")
        logger.info(f"{torch.cuda.get_device_capability(device)=}")
        logger.info(f"{torch.cuda.get_device_properties(device)=}")
    import torchaudio

    logger.info(f"{torchaudio.__version__=}")
    logger.info(f"{torchaudio.list_audio_backends()=}")


if __name__ == "__main__":
    app()
