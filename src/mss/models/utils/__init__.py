import logging
from functools import lru_cache
from logging import Logger


def parse_version(v_str: str) -> tuple[int, ...]:
    # e.g "2.1.0+cu118" -> (2, 1, 0)
    return tuple(map(int, v_str.split("+")[0].split(".")))


@lru_cache(10)
def log_once(logger: Logger, msg: object, *, level: int = logging.DEBUG) -> None:
    logger.log(level, msg)
