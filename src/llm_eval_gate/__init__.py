"""llm-eval-gate: evaluation gate for LLM features."""

__version__ = "0.1.0"

from . import metrics as _builtin_metrics  # noqa: E402,F401  (registers built-in metrics)
from .config import ConfigError, load_config, parse_config  # noqa: E402
from .registry import register  # noqa: E402
from .result import ExitCode, Generation, MetricOutcome, Severity  # noqa: E402
from .runner import run, score  # noqa: E402

__all__ = ["ConfigError", "ExitCode", "Generation", "MetricOutcome", "Severity", "load_config", "parse_config", "register", "run", "score"]
