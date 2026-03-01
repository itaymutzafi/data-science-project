"""Source code package for the Stock Market Prediction Pipeline.

Exposes the main subpackages and configuration.
"""

from . import config
from . import utils       # Utilities first (logging, stats, etc.)
from . import evaluation  # Depends on utils
from . import data        # Depends on evaluation (eda plots)
from . import features    # Depends on data, evaluation
from . import models      # Depends on all above

__all__ = [
    "config",
    "data",
    "features",
    "models",
    "evaluation",
    "utils",
]
