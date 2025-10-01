import logging
import sys


def set_up_logging(loglevel: int = logging.INFO):
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    return logging.getLogger(__name__)
