"""Matplotlib configuration for Chinese text support."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt


def configure_matplotlib_chinese():
    """Configure matplotlib to support Chinese characters in plots."""
    # Try to use Microsoft YaHei first, fallback to SimHei
    try:
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    except Exception:
        # If setting fails, just continue with default
        pass


# Auto-configure on import
configure_matplotlib_chinese()
