"""Dashboard configuration and setup utilities."""
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from config import Config


class DashboardConfig:
    """Handles dashboard configuration and page setup."""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_fetch_days = getattr(config, 'max_fetch_days', 90)
        
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="US Weather and Energy Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main header with improved styling
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0; margin-bottom: 2rem; 
                    background: linear-gradient(90deg, #1f77b4, #ff7f0e); 
                    border-radius: 10px; color: white;'>
            <h1 style='margin: 0; font-size: 2.5rem;'>🏙️ US Weather & Energy Dashboard</h1>
            <p style='margin: 0; font-size: 1.1rem; opacity: 0.9;'>Real-time analysis of energy consumption patterns across major US cities</p>
        </div>
        """, unsafe_allow_html=True)
    
    def setup_logging(self):
        """Configure dashboard-specific logging."""
        log_file = Path(self.config.data_paths['logs']) / f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"
        logger.add(
            log_file,
            rotation=self.config.logging.get('rotation', '10 MB'),
            retention=self.config.logging.get('retention', '7 days'),
            level=self.config.logging.get('level', 'INFO'),
            enqueue=True
        )


class SidebarControls:
    """Handles sidebar controls and user input."""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_fetch_days = getattr(config, 'max_fetch_days', 90)
        
    def setup_controls(self):
        """Setup sidebar controls and return user selections."""
        with st.sidebar:
            st.markdown("# 🎛️ Dashboard Controls")
            st.markdown("---")
            
            # Date Selection Section
            date_range = self._setup_date_selection()
            
            st.markdown("---")
            
            # City Selection Section
            selected_cities = self._setup_city_selection()
            
            st.markdown("---")
            
            # Display Options Section
            display_options = self._setup_display_options()
            
            st.markdown("---")
            
            # Advanced Settings Section
            advanced_settings = self._setup_advanced_settings()
            
            st.markdown("---")
            
            # Info Section
            self._show_info_section(selected_cities, date_range)
            
        return {
            'date_range': date_range,
            'selected_cities': selected_cities,
            'display_options': display_options,
            'advanced_settings': advanced_settings
        }
    
    def _setup_date_selection(self):
        """Setup date selection controls."""
        st.markdown("### 📅 Analysis Period")
        date_preset = st.selectbox(
            "Quick Select",
            options=["Last 7 days", "Last 14 days", "Last 30 days", "Last 60 days", "Last 90 days", "Custom Range"],
            index=2,
            help="Choose your analysis time period"
        )
        
        end_date = datetime.now().date() - timedelta(days=2)
        
        if date_preset == "Last 7 days":
            start_date = end_date - timedelta(days=7)
            return (start_date, end_date)
        elif date_preset == "Last 14 days":
            start_date = end_date - timedelta(days=14)
            return (start_date, end_date)
        elif date_preset == "Last 30 days":
            start_date = end_date - timedelta(days=30)
            return (start_date, end_date)
        elif date_preset == "Last 60 days":
            start_date = end_date - timedelta(days=60)
            return (start_date, end_date)
        elif date_preset == "Last 90 days":
            start_date = end_date - timedelta(days=90)
            return (start_date, end_date)
        else:  # Custom Range
            default_start = end_date - timedelta(days=30)
            return st.date_input(
                "Custom Date Range",
                value=(default_start, end_date),
                min_value=end_date - timedelta(days=self.max_fetch_days),
                max_value=end_date,
                help="Select custom start and end dates"
            )
    
    def _setup_city_selection(self):
        """Setup city selection controls."""
        st.markdown("### 🏙️ City Selection")
        
        all_cities = [city.name for city in self.config.cities]
        
        # Quick selection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Select All", use_container_width=True):
                st.session_state.selected_cities = all_cities
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.selected_cities = []
        
        # Multi-select for cities
        selected_cities = st.multiselect(
            "Choose Cities",
            options=all_cities,
            default=st.session_state.get('selected_cities', all_cities[:6]),
            help="Select cities for analysis - all visualizations will include selected cities",
            key='selected_cities'
        )
        
        if not selected_cities:
            st.warning("⚠️ Please select at least one city")
        
        return selected_cities
    
    def _setup_display_options(self):
        """Setup display options controls."""
        st.markdown("### ⚙️ Display Options")
        
        return {
            'show_correlations': st.checkbox("📊 Show Correlations", value=True),
            'show_last_day_change': st.checkbox("📈 Last Day Change", value=True),
            'show_quality': st.checkbox("🔍 Data Quality", value=False),
            'show_historical_table': st.checkbox("📋 Historical Data Table", value=True, 
                                                help="Show detailed historical data table")
        }
    
    def _setup_advanced_settings(self):
        """Setup advanced settings controls."""
        st.markdown("### 🔧 Advanced Settings")
        
        return {
            'lookback_days': st.slider("Historical Baseline (days)", 14, 90, 30),
            'recent_days': st.slider("Recent Period (days)", 1, 14, 7)
        }
    
    def _show_info_section(self, selected_cities, date_range):
        """Show info and current selections."""
        st.markdown("### ℹ️ About")
        st.info("This dashboard analyzes energy consumption patterns across major US cities, showing correlations with weather data and usage trends.")
        
        # Current selections summary
        if selected_cities:
            st.markdown("### 📋 Current Selection")
            st.write(f"**Cities:** {len(selected_cities)} selected")
            if hasattr(date_range, '__len__') and len(date_range) == 2:
                days_selected = (date_range[1] - date_range[0]).days
                st.write(f"**Period:** {days_selected} days")