"""Source code package for the Stock Market Prediction Pipeline.

Exposes the main subpackages and configuration.
"""

import os

from . import config
from . import data
from . import evaluation
from . import features
from . import models
from . import utils

# Pipelines are optional; import lazily to keep notebooks flexible.
_pipelines_dir = os.path.join(os.path.dirname(__file__), "pipelines")
if os.path.isdir(_pipelines_dir):
    try:
        from . import pipelines  # type: ignore
    except ImportError:
        pipelines = None
else:
    pipelines = None

__all__ = [
    "config",
    "data",
    "features",
    "models",
    "evaluation",
    "utils",
]

if "pipelines" in globals() and pipelines is not None:
    __all__.append("pipelines")
