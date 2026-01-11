"""Source code package for the Stock Market Prediction Pipeline.

Exposes the main subpackages and configuration.
"""

import os

from . import config
from . import utils       # Utilities first (logging, stats, etc.)
from . import evaluation  # Depends on utils
from . import data        # Depends on evaluation (eda plots)
from . import features    # Depends on data, evaluation
from . import models      # Depends on all above

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
