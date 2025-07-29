# __init__.py files for the dashboard modules

# dashboard/__init__.py
"""
US Weather and Energy Dashboard

A comprehensive Streamlit dashboard for analyzing energy consumption patterns
across major US cities, with correlations to weather data and usage trends.
"""

from .dashboard_core import Dashboard
from .ui_components import UIComponents
from .data_handlers import DataHandler
from .chart_generators import ChartGenerator
from .pipeline_manager import PipelineManager

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'Dashboard',
    'UIComponents', 
    'DataHandler',
    'ChartGenerator',
    'PipelineManager'
]

# dashboard/ui_components/__init__.py (if you want to further split UI components)
"""UI Components for the dashboard."""

# dashboard/data_handlers/__init__.py (if you want to further split data handlers)
"""Data handling and processing utilities."""

# dashboard/chart_generators/__init__.py (if you want to further split chart generators)
"""Chart generation and visualization utilities."""

# dashboard/pipeline_manager/__init__.py (if you want to further split pipeline management)
"""Pipeline management and Git operations."""