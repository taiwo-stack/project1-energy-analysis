"""Main dashboard class that orchestrates all components."""
import streamlit as st
import time
from datetime import datetime
from config import Config
from loguru import logger

# Import all the modular components
from visualization.dashboard_config import DashboardConfig, SidebarControls
from visualization.data_manager import DataManager, DataQualityChecker
from visualization.visualization import ChartGenerator
from visualization.table_display import HistoricalTableDisplay, SummaryMetrics
from visualization.pipeline_manager import PipelineChecker, PipelineOrchestrator


class Dashboard:
    """Main dashboard class that orchestrates all components."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize all components
        self.dashboard_config = DashboardConfig(config)
        self.sidebar_controls = SidebarControls(config)
        self.data_manager = DataManager(config)
        self.data_quality_checker = DataQualityChecker()
        self.chart_generator = ChartGenerator(self.data_manager)
        self.table_display = HistoricalTableDisplay(self.data_manager)
        self.summary_metrics = SummaryMetrics(self.data_manager)
        self.pipeline_orchestrator = PipelineOrchestrator()
        
        # Setup page and logging
        self.dashboard_config.setup_page()
        self.dashboard_config.setup_logging()
        
        # Get user controls
        self.controls = self.sidebar_controls.setup_controls()
        
        # Extract control values for easy access
        self.date_range = self.controls['date_range']
        self.selected_cities = self.controls['selected_cities']
        self.display_options = self.controls['display_options']
        self.advanced_settings = self.controls['advanced_settings']
    
    def display(self):
        """Main dashboard display method with improved layout."""
        # Check if cities are selected
        if not self.selected_cities:
            st.error("⚠️ Please select at least one city from the sidebar to proceed with the analysis.")
            return
        
        # Load and filter data
        with st.spinner("🔄 Loading data..."):
            data = self.data_manager.load_cached_data(self.date_range)
            filtered_data = self.data_manager.filter_data(data, self.selected_cities)
        
        if filtered_data.empty:
            st.warning("⚠️ No data available for the selected filters. Please adjust your selection in the sidebar.")
            return
        
        # Show summary metrics
        st.markdown("## 📊 Analysis Summary")
        self.summary_metrics.show_summary_metrics(
            filtered_data, 
            self.display_options['show_last_day_change']
        )
        
        # Show data quality if enabled
        self.data_quality_checker.show_data_quality(
            filtered_data, 
            self.display_options['show_quality'], 
            self.data_manager
        )
        
        # Show historical data table
        self.table_display.show_historical_data_table(
            filtered_data, 
            self.display_options['show_historical_table']
        )
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographic View", "📈 Time Series", "📊 Correlations", "🔥 Usage Patterns"])
        
        with tab1:
            st.markdown("### 🗺️ Geographic Energy Usage Overview")
            map_fig = self.chart_generator.create_geographic_map(
                filtered_data, 
                self.selected_cities,
                self.advanced_settings['lookback_days'],
                self.advanced_settings['recent_days'],
                self.display_options['show_last_day_change']
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
                self.display_options['show_correlations']
            )
            st.plotly_chart(corr_fig, use_container_width=True)
        
        with tab4:
            st.markdown("### 🔥 Energy Usage Patterns by Temperature & Day")
            st.info(f"📋 Displaying heatmap for {len(self.selected_cities)} selected cities with day-of-week labels")
            heatmap_fig = self.chart_generator.create_heatmap(filtered_data, self.selected_cities)
            st.plotly_chart(heatmap_fig, use_container_width=True)


class DashboardManager:
    """High-level dashboard manager that handles pipeline integration."""
    
    def __init__(self):
        self.config = Config.load()
        self.pipeline_orchestrator = PipelineOrchestrator()
    
    def run(self):
        """Main entry point for the dashboard with pipeline management."""
        # Initialize session state for message timing
        if 'message_timestamp' not in st.session_state:
            st.session_state.message_timestamp = None
        if 'show_success_message' not in st.session_state:
            st.session_state.show_success_message = False
        
        # Handle manual refresh trigger
        if st.session_state.get('manual_refresh', False):
            st.session_state.manual_refresh = False  # Reset the trigger
            success = self.pipeline_orchestrator.handle_pipeline_execution(is_manual=True)
            
        # Check if pipeline should run automatically (only if data is 3+ days old)
        elif PipelineChecker.should_run_pipeline(self.config):
            success = self.pipeline_orchestrator.handle_pipeline_execution(is_manual=False)
        
        # Check if success message should be hidden after 5 seconds
        if (st.session_state.show_success_message and 
            st.session_state.message_timestamp and 
            time.time() - st.session_state.message_timestamp > 5):
            st.session_state.show_success_message = False
            st.session_state.message_timestamp = None
            # Clear any cached data to ensure fresh data is loaded
            st.cache_data.clear()
            st.rerun()  # Refresh to hide the message and reload with fresh data
        
        # Show data status and manual refresh option (only if no success message is showing)
        if not st.session_state.show_success_message:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("📊 Using existing data >>> You can refresh here >> Note:: up 10min to complete")
            with col2:
                if st.button("🔄 Refresh Now", key="manual_refresh_btn", help="Manually refresh data pipeline"):
                    st.session_state.manual_refresh = True
                    st.rerun()
        
        # Initialize and display the main dashboard
        try:
            dashboard = Dashboard(self.config)
            dashboard.display()
        except Exception as e:
            st.error("❌ Failed to initialize dashboard. Please check the logs.")
            logger.critical(f"Dashboard failed: {str(e)}")


def main():
    """Main entry point for the dashboard application."""
    dashboard_manager = DashboardManager()
    dashboard_manager.run()


if __name__ == "__main__":
    main()