"""Neural POS Tagger with Discrete Representation Learning."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config.config import Config, ExperimentConfig
from .experiments.experiment_runner import ExperimentRunner

__all__ = ["Config", "ExperimentConfig", "ExperimentRunner"]
