#!/usr/bin/env python3
"""Run the complete trial-order analysis and figure workflow.

Usage: python main.py [config] [--refresh | --from-saved]
"""

import sys
from utils.constants import SCRIPT_VERSION
from custom_logger import configure_logging, logger


def main() -> int:
    """Keep version reporting lightweight; delegate the workflow to the package."""
    if sys.argv[1:] == ["--version"]:
        configure_logging()
        logger.info("main.py %s", SCRIPT_VERSION)
        return 0
    from utils.cli import main as run
    return run()


if __name__ == "__main__":
    sys.exit(main())
