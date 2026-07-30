"""Reading the YAML experiment configurations."""

from __future__ import annotations

import yaml


def load_config(path):
    """Load a configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)
