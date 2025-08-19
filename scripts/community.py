"""Utils to collate all models trained by the community."""

import io
import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Annotated, Generator, Iterator, Sequence

import httpx
import typer
from rich.logging import RichHandler

from splifft import PATH_SCRIPTS

PATH_TMP = PATH_SCRIPTS / "tmp"
URL_JARREDOU_MSST_COLAB = "https://raw.githubusercontent.com/jarredou/Music-Source-Separation-Training-Colab-Inference/refs/heads/main/Music_Source_Separation_Training_(Colab_Inference).ipynb"
PATH_JARREDOU_MSST_COLAB = PATH_TMP / "jarredou_msst_colab.ipynb"
PATH_JARREDOU_MSST_JSON = PATH_TMP / "jarredou_msst_colab.json"

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)
app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@dataclass
class JarredouModel:
    name: str
    model_type: str
    config_path: str
    checkpoint_path: str
    config_url: str
    checkpoint_url: str

OutputPath = Annotated[Path, typer.Option("--output", "-o", help="use `-` for stdout")]

@app.command()
def parse_jarredou_colab(
    url: str = URL_JARREDOU_MSST_COLAB,
    path_ipynb: Path = PATH_JARREDOU_MSST_COLAB,
    output_path: OutputPath = PATH_JARREDOU_MSST_JSON,
) -> None:
    """Fetch and parse models from jarredou's colab notebook."""
    if not path_ipynb.exists():
        logger.info(f"downloading {url=}...")
        path_ipynb.parent.mkdir(exist_ok=True, parents=True)
        response = httpx.get(url)
        response.raise_for_status()
        path_ipynb.write_bytes(response.content)
        logger.info(f"wrote {path_ipynb=}")

    pattern = re.compile(
        r"(?:el)?if model == '(?P<name>[^']+)':\s*\n"
        r"\s*model_type = '(?P<model_type>[^']+)'.*\n"
        r"\s*config_path = '(?P<config_path>[^']+)'.*\n"
        r"\s*start_check_point = '(?P<checkpoint_path>[^']+)'.*\n"
        r"\s*download_file\('(?P<config_url>[^']+)'\).*\n"
        r"\s*download_file\('(?P<checkpoint_url>[^']+)'\)",
        re.MULTILINE,
    )
    models = []
    for block in _colab_blocks(path_ipynb):
        for match in pattern.finditer(block):
            models.append(asdict(JarredouModel(**match.groupdict())))

    output_json = json.dumps(models, indent=4)
    if str(output_path) == "-":
        print(output_json)
    else:
        output_path.write_text(output_json, encoding="utf-8")
        logger.info(f"wrote {len(models)} models to {output_path}")


def _colab_blocks(path_ipynb: Path) -> Generator[str, None, None]:
    with open(path_ipynb, "r", encoding="utf-8") as f:
        ipynb = json.load(f)
    for cell in ipynb.get("cells", []):
        source_lines = cell.get("source", [])
        if cell.get("cell_type") != "code":
            continue
        yield "\n".join(
            li
            for line in source_lines
            if (not (li := line.strip()).startswith("%")) and not li.startswith("!")
        )


PATH_GUIDE_DOCX = PATH_TMP / "guide.docx"
PATH_GUIDE_MD = PATH_TMP / "guide.md"
PATH_GUIDE_MODELS_MD = PATH_TMP / "guide_filtered.md"
# fmt: off
KEYWORDS_GUIDE_NON_MODELS = [
    "*plugins*", "mdx settings", "uvr5 gui", "sources of flacs",
    "arigato78", "karafan", "ensemble", "ripple", "training", "tips",
    "troubleshooting", "stems/multitracks", "gpu acceleration",
    "audioshake", "lalal", "gsep", "dango", "moises",
    "dolby atmos ripping", "ai-killing"
]
# fmt: on

@app.command()
def parse_guide(
    blacklist_keywords: list[str] = KEYWORDS_GUIDE_NON_MODELS,
    output_path: OutputPath = PATH_GUIDE_MODELS_MD
) -> None:
    """Output a pruned version of the guide in markdown."""
    if not PATH_GUIDE_DOCX.exists():
        logger.error(
            f"{PATH_GUIDE_DOCX=} not found, "
            "navigate to https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c/edit, "
            "then `File > Download > Microsoft Word (.docx)`"
        )
        return
    subprocess.run(["pandoc", str(PATH_GUIDE_DOCX), "-o", str(PATH_GUIDE_MD)])
    def write_pruned_guide(file_out: IO[str]) -> None:
        with open(PATH_GUIDE_MD, "r", encoding="utf-8") as file_in:
            for line in prune_sections(file_in, blacklist_keywords):
                file_out.write(line)

    if str(output_path) == "-":
        buf = io.StringIO()
        write_pruned_guide(buf)
        print(buf.getvalue())
        return
    with open(output_path, "w", encoding="utf-8") as fout:
        write_pruned_guide(fout)


def prune_sections(
    lines: Iterator[str], blacklist_keywords: Sequence[str]
) -> Generator[str, None, None]:
    exclude_level: int | None = None
    used_keywords = []
    for line in lines:
        if (line_ls := line.lstrip().lower()).startswith("#"):
            # print(line_ls)
            curr_level = line_ls.split(" ")[0].count("#")
            if exclude_level is not None and curr_level <= exclude_level:
                exclude_level = None
            for keyword in blacklist_keywords:
                if keyword not in line_ls:
                    continue
                exclude_level = curr_level
                used_keywords.append(keyword)
                logger.info(f"excluding `{line_ls}`")
        if exclude_level is None:
            yield line
    if (unused := set(blacklist_keywords) - set(used_keywords)):
        logger.warning(f"unused keywords: {unused}")


if __name__ == "__main__":
    app()
