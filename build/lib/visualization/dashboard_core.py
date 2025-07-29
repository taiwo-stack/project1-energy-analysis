"""Core dashboard class with main functionality."""
import streamlit as st
from streamlit.runtime.caching import cache_data
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from config import Config
from analysis import Analyzer
from loguru import logger
from pathlib import Path

from .ui_components import UIComponents
from .data_handlers import DataHandler
from .chart_generators import ChartGenerator
from .pipeline_manager import PipelineManager


class Dashboard(Analyzer):
    """Interactive dashboard extending Analyzer for visualizations."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        # Initialize components
        self.ui = UIComponents()
        self.data_handler = DataHandler(config)
        self.chart_generator = ChartGenerator()
        self.pipeline_manager = PipelineManager(config)
        
        self._setup_page()
        self._setup_controls()
        self._setup_logging()
    
    def _setup_page(self):
        """Configure Streamlit page settings."""
        self.ui.setup_page_config()
        self.ui.render_main_header()
    
    def _setup_controls(self):
        """Setup sidebar controls."""
        controls = self.ui.setup_sidebar_controls(self.config)
        
        # Extract control values
        self.date_range = controls['date_range']
        self.selected_cities = controls['selected_cities']
        self.show_correlations = controls['show_correlations']
        self.show_last_day_change = controls['show_last_day_change']
        self.show_quality = controls['show_quality']
        self.show_historical_table = controls['show_historical_table']
        self.lookback_days = controls['lookback_days']
        self.recent_days = controls['recent_days']
    
    def _setup_logging(self):
        """Configure dashboard-specific logging."""
        log_file = Path(self.config.data_paths['logs']) / f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"
        logger.add(
            log_file,
            rotation=self.config.logging.get('rotation', '10 MB'),
            retention=self.config.logging.get('retention', '7 days'),
            level=self.config.logging.get('level', 'INFO'),
            enqueue=True
        )
    
    @cache_data(ttl=3600, show_spinner="🔄 Loading data...")
    def _load_data(_self, date_range: Tuple[datetime.date, datetime.date]) -> pd.DataFrame:
        """Load and cache processed data using Analyzer."""
        return _self.load_data(date_range)
    
    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on user selections."""
        return self.data_handler.filter_data(df, self.selected_cities)
    
    def display(self):
        """Main dashboard display method with improved layout."""
        # Check if cities are selected
        if not self.selected_cities:
            st.error("⚠️ Please select at least one city from the sidebar to proceed with the analysis.")
            return
        
        # Handle pipeline management
        if self.pipeline_manager.should_run_pipeline():
            self.pipeline_manager.handle_pipeline_execution()
        
        # Load and filter data
        with st.spinner("🔄 Loading data..."):
            data = self._load_data(self.date_range)
            filtered_data = self._filter_data(data)
        
        if filtered_data.empty:
            st.warning("⚠️ No data available for the selected filters. Please adjust your selection in the sidebar.")
            return
        
        # Show summary metrics
        st.markdown("## 📊 Analysis Summary")
        self.ui.show_summary_metrics(filtered_data, self.show_last_day_change, self.calculate_last_day_change)
        
        # Show data quality if enabled
        if self.show_quality:
            self.ui.show_data_quality(filtered_data, self.generate_data_quality_report, self.data_handler.check_data_freshness)
        
        # Show historical data table
        if self.show_historical_table:
            self.ui.show_historical_data_table(filtered_data)
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographic View", "📈 Time Series", "📊 Correlations", "🔥 Usage Patterns"])
        
        with tab1:
            st.markdown("### 🗺️ Geographic Energy Usage Overview")
            map_fig = self.chart_generator.create_geographic_map(
                filtered_data, 
                self.calculate_usage_levels,
                self.calculate_last_day_change if self.show_last_day_change else None,
                self.selected_cities,
                self.lookback_days,
                self.recent_days
            )
            st.plotly_chart(map_fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📈 Temperature & Energy Timeline")
            ts_fig = self.chart_generator.create_time_series(filtered_data, self.selected_cities)
            st.plotly_chart(ts_fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 📊 Temperature vs Energy Correlation")
            corr_fig = self.chart_generator.create_correlation_plot(
                filtered_data, 
                self.selected_cities,
                self.calculate_regression,
                self.calculate_correlations,
                self.show_correlations
            )
            st.plotly_chart(corr_fig, use_container_width=True)
        
        with tab4:
            st.markdown("### 🔥 Energy Usage Patterns by Temperature & Day")
            st.info(f"📋 Displaying heatmap for {len(self.selected_cities)} selected cities with day-of-week labels")
            heatmap_fig = self.chart_generator.create_heatmap(filtered_data, self.selected_cities)
            st.plotly_chart(heatmap_fig, use_container_width=True)