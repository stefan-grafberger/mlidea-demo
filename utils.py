import os
import random
from pathlib import Path

import numpy
from fairlearn.metrics import MetricFrame


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent

def get_open_ai_key():
    return ""

def metrics_to_str(metrics):
    if isinstance(metrics, list):
        pretty_metrics = []
        for x in metrics:
            if isinstance(x, float):
                pretty_metrics.append(f'{x:.2f}')
            elif isinstance(x, MetricFrame):
                pretty_metrics.append(str(x.by_group))
        return "[" + ', '.join(pretty_metrics) + "]"
    elif isinstance(metrics, float):
        return f"{metrics:.2f}"
    elif isinstance(metrics, str):
        return metrics
    else:
        raise TypeError()