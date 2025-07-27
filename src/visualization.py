"""Interactive Streamlit dashboard for weather and energy visualizations."""

import streamlit as st
from streamlit.runtime.caching import cache_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from config import Config
from analysis import Analyzer
from loguru import logger
from pathlib import Path

class Dashboard(Analyzer):
    """Interactive dashboard extending Analyzer for visualizations."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self._setup_page()
        self._setup_controls()
        self._setup_logging()
    
    def _setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="US Weather and Energy Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"  # Changed to expanded for sidebar controls
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
    
    
    def _setup_controls(self):
        """Setup sidebar controls."""
        with st.sidebar:
            st.markdown("# 🎛️ Dashboard Controls")
            st.markdown("---")
            
            # Date Selection Section
            st.markdown("### 📅 Analysis Period")
            # Simplified date selection with presets
            date_preset = st.selectbox(
                "Quick Select",
                options=["Last 7 days", "Last 14 days", "Last 30 days", "Last 60 days", "Last 90 days", "Custom Range"],
                index=2,  # Default to 30 days
                help="Choose your analysis time period"
            )
            
            # Calculate date range based on preset - Updated to use 2-day buffer
            end_date = datetime.now().date() - timedelta(days=2)  # Fixed 2-day buffer
            if date_preset == "Last 7 days":
                start_date = end_date - timedelta(days=7)
                self.date_range = (start_date, end_date)
            elif date_preset == "Last 14 days":
                start_date = end_date - timedelta(days=14)
                self.date_range = (start_date, end_date)
            elif date_preset == "Last 30 days":
                start_date = end_date - timedelta(days=30)
                self.date_range = (start_date, end_date)
            elif date_preset == "Last 60 days":
                start_date = end_date - timedelta(days=60)
                self.date_range = (start_date, end_date)
            elif date_preset == "Last 90 days":
                start_date = end_date - timedelta(days=90)
                self.date_range = (start_date, end_date)
            else:  # Custom Range
                default_start = end_date - timedelta(days=30)
                self.date_range = st.date_input(
                    "Custom Date Range",
                    value=(default_start, end_date),
                    min_value=end_date - timedelta(days=self.max_fetch_days),
                    max_value=end_date,
                    help="Select custom start and end dates"
                )
            
            st.markdown("---")
            
            # City Selection Section
            st.markdown("### 🏙️ City Selection")
            
            # All available cities
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
            self.selected_cities = st.multiselect(
                "Choose Cities",
                options=all_cities,
                default=st.session_state.get('selected_cities', all_cities[:6]),  # Default to first 6
                help="Select cities for analysis - all visualizations will include selected cities",
                key='selected_cities'
            )
            
            if not self.selected_cities:
                st.warning("⚠️ Please select at least one city")
            
            st.markdown("---")
            
            # Display Options Section
            st.markdown("### ⚙️ Display Options")
            
            # Main display options
            self.show_correlations = st.checkbox("📊 Show Correlations", value=True)
            self.show_last_day_change = st.checkbox("📈 Last Day Change", value=True)
            self.show_quality = st.checkbox("🔍 Data Quality", value=False)
            self.show_historical_table = st.checkbox("📋 Historical Data Table", value=True, help="Show detailed historical data table")
            
            st.markdown("---")
            
            # Advanced Settings Section
            st.markdown("### 🔧 Advanced Settings")
            self.lookback_days = st.slider("Historical Baseline (days)", 14, 90, 30)
            self.recent_days = st.slider("Recent Period (days)", 1, 14, 7)
            
            st.markdown("---")
            
            # Info Section
            st.markdown("### ℹ️ About")
            st.info("This dashboard analyzes energy consumption patterns across major US cities, showing correlations with weather data and usage trends.")
            
            # Current selections summary
            if self.selected_cities:
                st.markdown("### 📋 Current Selection")
                st.write(f"**Cities:** {len(self.selected_cities)} selected")
                if hasattr(self, 'date_range') and len(self.date_range) == 2:
                    days_selected = (self.date_range[1] - self.date_range[0]).days
                    st.write(f"**Period:** {days_selected} days")

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
        if self.selected_cities:
            df = df[df['city'].isin(self.selected_cities)]
        return df
    
    def _prepare_historical_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and format data for the historical table display."""
        try:
            if df.empty:
                return pd.DataFrame()
            
            # Create a formatted version of the dataframe for display
            display_df = df.copy()
            
            # Sort by date (newest first) and city
            display_df = display_df.sort_values(['date', 'city'], ascending=[False, True])
            
            # Format columns for better display
            formatted_data = []
            for _, row in display_df.iterrows():
                formatted_row = {
                    '📅 Date': row['date'].strftime('%Y-%m-%d'),
                    '🏙️ City': row['city'],
                    '🌡️ Temp (°F)': f"{row.get('temperature_avg', row.get('temperature_max', 0)):.1f}",
                    '⚡ Energy (MWh)': f"{row['energy_demand']:,.0f}",
                    '📊 Day': row['day_of_week'],
                    '📈 Weekend': '✅ Yes' if row.get('is_weekend', False) else '❌ No',
                }
                
                # Add temperature range if available
                if 'temperature_min' in row and 'temperature_max' in row:
                    formatted_row['🌡️ Min/Max'] = f"{row['temperature_min']:.1f} / {row['temperature_max']:.1f}"
                
                # Add weather description if available
                if 'weather_description' in row and pd.notna(row['weather_description']):
                    formatted_row['🌤️ Weather'] = row['weather_description'].title()
                
                formatted_data.append(formatted_row)
            
            return pd.DataFrame(formatted_data)
            
        except Exception as e:
            logger.error(f"Failed to prepare historical table data: {str(e)}")
            return pd.DataFrame()
    
    def _show_historical_data_table(self, df: pd.DataFrame):
        """Display enhanced historical data table with collapsible interface."""
        if not self.show_historical_table or df.empty:
            return
        
        # Prepare data for table
        table_data = self._prepare_historical_table_data(df)
        
        if table_data.empty:
            st.warning("⚠️ No historical data available for table display")
            return
        
        # Create collapsible section with custom styling
        st.markdown("---")
        
        # Custom CSS for the table
        st.markdown("""
        <style>
        .historical-table-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .table-header {
            color: white;
            text-align: center;
            margin-bottom: 15px;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .table-stats {
            display: flex;
            justify-content: space-around;
            margin-bottom: 15px;
            color: white;
            font-size: 0.9rem;
        }
        .stat-item {
            text-align: center;
            padding: 8px 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }
        .dataframe {
            font-size: 0.85rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create expandable section
        with st.expander("📋 **Historical Data Explorer** - Detailed Records & Analysis", expanded=False):
            # Add container with custom styling
            st.markdown('<div class="historical-table-container">', unsafe_allow_html=True)
            
            # Table header
            st.markdown('<div class="table-header">📊 Historical Weather & Energy Data</div>', unsafe_allow_html=True)
            
            # Summary statistics
            total_records = len(table_data)
            date_range_str = f"{table_data['📅 Date'].iloc[-1]} to {table_data['📅 Date'].iloc[0]}"
            cities_count = len(table_data['🏙️ City'].unique())
            
            st.markdown(f"""
            <div class="table-stats">
                <div class="stat-item">
                    <div><strong>📊 Total Records</strong></div>
                    <div>{total_records:,}</div>
                </div>
                <div class="stat-item">
                    <div><strong>🏙️ Cities</strong></div>
                    <div>{cities_count}</div>
                </div>
                <div class="stat-item">
                    <div><strong>📅 Date Range</strong></div>
                    <div>{date_range_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add search and filter options
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_term = st.text_input(
                    "🔍 Search cities or weather conditions:",
                    placeholder="Type to filter records...",
                    help="Search by city name or weather description"
                )
            
            with col2:
                sort_by = st.selectbox(
                    "📊 Sort by:",
                    options=['📅 Date', '🏙️ City', '⚡ Energy (MWh)', '🌡️ Temp (°F)'],
                    index=0,
                    help="Choose sorting column"
                )
            
            with col3:
                sort_order = st.selectbox(
                    "🔄 Order:",
                    options=['Descending', 'Ascending'],
                    index=0,
                    help="Sort order"
                )
            
            # Apply search filter
            filtered_table = table_data.copy()
            if search_term:
                mask = (
                    filtered_table['🏙️ City'].str.contains(search_term, case=False, na=False) |
                    (filtered_table.get('🌤️ Weather', pd.Series(dtype='object')).str.contains(search_term, case=False, na=False))
                )
                filtered_table = filtered_table[mask]
            
            # Apply sorting
            if sort_by in filtered_table.columns:
                ascending = (sort_order == 'Ascending')
                if sort_by in ['⚡ Energy (MWh)', '🌡️ Temp (°F)']:
                    # Convert to numeric for proper sorting
                    sort_values = pd.to_numeric(filtered_table[sort_by].str.replace(',', ''), errors='coerce')
                    filtered_table = filtered_table.iloc[sort_values.sort_values(ascending=ascending).index]
                else:
                    filtered_table = filtered_table.sort_values(sort_by, ascending=ascending)
            
            # Display record count after filtering
            if len(filtered_table) != len(table_data):
                st.info(f"📊 Showing {len(filtered_table):,} of {len(table_data):,} records (filtered)")
            
            # Display the table with fixed height and scrolling
            if not filtered_table.empty:
                # Configure table display options
                st.markdown("### 📋 Data Table")
                st.markdown("*Scroll within the table to view all records*")
                
                # Display table with pagination-like behavior
                st.dataframe(
                    filtered_table,
                    use_container_width=True,
                    hide_index=True,
                    height=400,  # Fixed height for scrolling (approximately 14 rows)
                    column_config={
                        '📅 Date': st.column_config.TextColumn('📅 Date', width='medium'),
                        '🏙️ City': st.column_config.TextColumn('🏙️ City', width='medium'),
                        '🌡️ Temp (°F)': st.column_config.TextColumn('🌡️ Temp (°F)', width='small'),
                        '⚡ Energy (MWh)': st.column_config.TextColumn('⚡ Energy (MWh)', width='medium'),
                        '📊 Day': st.column_config.TextColumn('📊 Day', width='small'),
                        '📈 Weekend': st.column_config.TextColumn('📈 Weekend', width='small'),
                    }
                )
                
                # Add download option
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    # Export to CSV
                    csv_data = filtered_table.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"historical_energy_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        help="Download filtered data as CSV file"
                    )
                
                with col2:
                    # Quick stats toggle
                    if st.button("📊 Quick Stats", help="Show summary statistics"):
                        st.session_state.show_quick_stats = not st.session_state.get('show_quick_stats', False)
                
                with col3:
                    st.markdown(f"*📊 Table showing {len(filtered_table):,} records with 400px height for scrolling*")
                
                # Show quick stats if toggled
                if st.session_state.get('show_quick_stats', False):
                    st.markdown("#### 📊 Quick Statistics")
                    
                    # Extract numeric values for statistics - handle NaN values properly
                    energy_values = pd.to_numeric(filtered_table['⚡ Energy (MWh)'].str.replace(',', ''), errors='coerce').dropna()
                    temp_values = pd.to_numeric(filtered_table['🌡️ Temp (°F)'], errors='coerce').dropna()
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        avg_energy = energy_values.mean() if len(energy_values) > 0 else 0
                        st.metric("⚡ Avg Energy", f"{avg_energy:,.0f} MWh")
                    with stat_col2:
                        avg_temp = temp_values.mean() if len(temp_values) > 0 else 0
                        st.metric("🌡️ Avg Temp", f"{avg_temp:.1f}°F")
                    with stat_col3:
                        max_energy = energy_values.max() if len(energy_values) > 0 else 0
                        st.metric("📈 Max Energy", f"{max_energy:,.0f} MWh")
                    with stat_col4:
                        max_temp = temp_values.max() if len(temp_values) > 0 else 0
                        st.metric("🌡️ Max Temp", f"{max_temp:.1f}°F")
            
            else:
                st.warning("⚠️ No records match your search criteria. Try adjusting your filters.")
    
    def calculate_last_day_change(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate last recorded day's energy usage % change from the previous day for all cities."""
        try:
            if df.empty:
                return {}
            
            changes = {}
            
            for city in df['city'].unique():
                city_df = df[df['city'] == city].sort_values('date')
                
                if len(city_df) < 2:
                    continue
                
                # Get the last two available data points for this city
                last_two = city_df.tail(2)
                
                if len(last_two) == 2:
                    latest_usage = last_two.iloc[-1]['energy_demand']
                    previous_usage = last_two.iloc[-2]['energy_demand']
                    latest_date = last_two.iloc[-1]['date']
                    previous_date = last_two.iloc[-2]['date']
                    
                    if previous_usage != 0:
                        pct_change = ((latest_usage - previous_usage) / previous_usage) * 100
                        days_between = (latest_date - previous_date).days
                        
                        changes[city] = {
                            'latest_usage': round(latest_usage, 2),
                            'previous_usage': round(previous_usage, 2),
                            'pct_change': round(pct_change, 1),
                            'latest_date': latest_date.strftime('%Y-%m-%d'),
                            'previous_date': previous_date.strftime('%Y-%m-%d'),
                            'days_between': days_between
                        }
            
            return changes
            
        except Exception as e:
            logger.error(f"Failed to calculate last day's change: {str(e)}")
            return {}
    
    def create_heatmap(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """Create improved heatmap with all selected cities and visible day labels."""
        try:
            if df.empty:
                st.warning("⚠️ No data for heatmap")
                return go.Figure()
            
            df = df.copy()
            if 'temperature_avg' not in df.columns:
                df['temperature_avg'] = df.get('temperature_max', df.get('temperature_min', np.nan))
            
            cities_to_plot = [city for city in selected_cities if city in df['city'].unique()]
            
            if not cities_to_plot:
                st.warning("⚠️ No valid cities selected for heatmap")
                return go.Figure()
            
            # Create temperature bins
            bins = [-float('inf'), 50, 60, 70, 80, 90, float('inf')]
            labels = ['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
            df['temp_range'] = pd.cut(df['temperature_avg'], bins=bins, labels=labels, include_lowest=True)
            
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            temp_order = labels
            
            # Dynamic subplot layout calculation
            num_cities = len(cities_to_plot)
            if num_cities == 1:
                rows, cols = 1, 1
            elif num_cities == 2:
                rows, cols = 1, 2
            elif num_cities <= 4:
                rows, cols = 2, 2
            elif num_cities <= 6:
                rows, cols = 2, 3
            elif num_cities <= 9:
                rows, cols = 3, 3
            elif num_cities <= 12:
                rows, cols = 3, 4
            else:
                rows, cols = 4, 4
                cities_to_plot = cities_to_plot[:16]  # Limit to 16 for display
                st.info(f"ℹ️ Showing heatmap for first 16 cities: {', '.join(cities_to_plot)}")
            
            # Create subplots with better spacing
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"<b>{city}</b>" for city in cities_to_plot],
                vertical_spacing=0.12,
                horizontal_spacing=0.08
            )
            
            # Track min/max values for consistent color scale
            all_values = []
            
            for idx, city in enumerate(cities_to_plot):
                row = (idx // cols) + 1
                col = (idx % cols) + 1
                
                city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand', 'temp_range', 'day_of_week'])
                
                if len(city_df) < 5:  # Require minimum data for meaningful heatmap
                    # Add empty heatmap with message
                    empty_z = np.full((len(temp_order), len(days_order)), np.nan)
                    fig.add_trace(
                        go.Heatmap(
                            z=empty_z,
                            x=days_order,
                            y=temp_order,
                            showscale=False,
                            hoverinfo='skip',
                            colorscale='RdYlBu_r'
                        ),
                        row=row, col=col
                    )
                    fig.add_annotation(
                        text=f"Insufficient data<br>({len(city_df)} records)",
                        x=0.5, y=0.5,
                        xref=f"x{idx+1}", yref=f"y{idx+1}",
                        showarrow=False,
                        font=dict(size=12, color="red")
                    )
                    continue
                
                # Create heatmap data - fix the ambiguous array issue
                try:
                    city_heatmap = city_df.groupby(['temp_range', 'day_of_week'], observed=True)['energy_demand'].mean().unstack(fill_value=np.nan)
                    city_heatmap = city_heatmap.reindex(index=temp_order, columns=days_order, fill_value=np.nan)
                except Exception as group_error:
                    logger.warning(f"Groupby failed for {city}: {str(group_error)}")
                    # Fallback: create empty heatmap
                    city_heatmap = pd.DataFrame(
                        data=np.full((len(temp_order), len(days_order)), np.nan),
                        index=temp_order,
                        columns=days_order
                    )
                
                # Collect values for color scale
                valid_values = city_heatmap.values[~np.isnan(city_heatmap.values)]
                if len(valid_values) > 0:
                    all_values.extend(valid_values)
                
                # FIXED: Prepare text for display - Handle NaN values properly without casting warnings
                text_values = np.where(
                    np.isnan(city_heatmap.values), 
                    '', 
                    # Use vectorized approach to avoid casting NaN to int
                    np.vectorize(lambda x: str(int(round(x))) if not np.isnan(x) else '')(city_heatmap.values)
                )
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=city_heatmap.values,
                        x=city_heatmap.columns,
                        y=city_heatmap.index,
                        colorscale='RdYlBu_r',
                        text=text_values,
                        texttemplate='%{text}',
                        textfont=dict(size=8),
                        showscale=(idx == 0),  # Show colorbar only for first subplot
                        colorbar=dict(
                            title='Energy (MWh)',
                            x=1.02,
                            len=0.8
                        ) if idx == 0 else None,
                        hovertemplate=(
                            f'<b>{city}</b><br>'
                            'Day: %{x}<br>'
                            'Temp Range: %{y}<br>'
                            'Energy: %{z:,.0f} MWh<br>'
                            '<extra></extra>'
                        )
                    ),
                    row=row, col=col
                )
            
            # Set consistent color range for all heatmaps
            if all_values:
                zmin, zmax = min(all_values), max(all_values)
                # Update all heatmap traces with consistent color range
                for trace in fig.data:
                    if hasattr(trace, 'zmin'):
                        trace.update(zmin=zmin, zmax=zmax)
            
            # Calculate dynamic height based on number of rows
            base_height = 250
            fig_height = max(base_height * rows, 400)
            
            # Update layout with better styling
            fig.update_layout(
                title=dict(
                    text=f'🔥 Energy Usage Patterns by Temperature & Day ({num_cities} Cities)',
                    x=0.5,
                    font=dict(size=18, color='#1f77b4')
                ),
                height=fig_height,
                margin=dict(l=50, r=100, t=60, b=40),
                font=dict(size=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Show day labels on ALL subplots for better readability
            for i in range(len(cities_to_plot)):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                # Show x-axis (days) on all subplots
                fig.update_xaxes(
                    showticklabels=True,
                    tickangle=45,
                    tickfont=dict(size=8),
                    row=row, col=col
                )
                
                # Show y-axis (temperature) on left column only
                if col == 1:
                    fig.update_yaxes(
                        showticklabels=True,
                        tickfont=dict(size=8),
                        row=row, col=col
                    )
                else:
                    fig.update_yaxes(showticklabels=False, row=row, col=col)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create heatmap: {str(e)}")
            st.error(f"❌ Heatmap failed: {str(e)}")
            return go.Figure()
    
    def create_correlation_plot(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """Create enhanced scatter plot with regression line."""
        try:
            if df.empty:
                st.warning("⚠️ No data for correlation plot")
                return go.Figure()
            
            df = df.copy()
            if 'temperature_avg' not in df.columns:
                df['temperature_avg'] = df.get('temperature_max', df.get('temperature_min', np.nan))
            
            fig = go.Figure()
            regression_stats = self.calculate_regression(df, selected_cities)
            correlations = self.calculate_correlations(df, selected_cities)
            
            colors = px.colors.qualitative.Set1
            
            for idx, city in enumerate(selected_cities):
                if city not in df['city'].unique():
                    continue
                    
                city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand'])
                if city_df.empty:
                    continue
                
                color = colors[idx % len(colors)]
                
                # Add scatter points
                fig.add_trace(go.Scatter(
                    x=city_df['temperature_avg'],
                    y=city_df['energy_demand'],
                    mode='markers',
                    name=f'{city}',
                    marker=dict(color=color, size=6, opacity=0.7),
                    text=city_df['date'].dt.strftime('%Y-%m-%d'),
                    hovertemplate=(
                        f'<b>{city}</b><br>'
                        'Date: %{text}<br>'
                        'Temperature: %{x:.1f}°F<br>'
                        'Energy: %{y:,.0f} MWh<br>'
                        '<extra></extra>'
                    )
                ))
                
                # Add regression line
                stats = regression_stats.get(city, {})
                if not np.isnan(stats.get('slope', np.nan)):
                    x_range = [city_df['temperature_avg'].min(), city_df['temperature_avg'].max()]
                    y_pred = [stats['slope'] * x + stats['intercept'] for x in x_range]
                    
                    corr_value = correlations.get(city, np.nan)
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name=f'{city} Trend (r={corr_value:.3f})',
                        line=dict(color=color, dash='dash', width=2),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>{city} Regression</b><br>'
                            f'Slope: {stats["slope"]:.2f}<br>'
                            f'R²: {stats["r_squared"]:.3f}<br>'
                            f'Correlation: {corr_value:.3f}<br>'
                            '<extra></extra>'
                        )
                    ))
            
            # Enhanced correlation annotation
            if self.show_correlations and correlations:
                corr_text = "<b>📊 Correlations (r):</b><br>"
                sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
                
                for city, corr in sorted_corr:
                    if not np.isnan(corr):
                        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                        corr_text += f"• {city}: {corr:.3f} ({strength})<br>"
                
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=corr_text,
                    showarrow=False,
                    bgcolor='rgba(0, 0, 0, 0.95)',
                    bordercolor='#1f77b4',
                    borderwidth=2,
                    align='left',
                    font=dict(size=10)
                )
            
            fig.update_layout(
                title=dict(
                    text='📊 Temperature vs. Energy Demand Correlation',
                    x=0.5,
                    font=dict(size=18, color='#1f77b4')
                ),
                xaxis_title='🌡️ Average Temperature (°F)',
                yaxis_title='⚡ Energy Demand (MWh)',
                height=500,
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                hovermode='closest',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.5)'
            )
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create correlation plot: {str(e)}")
            st.error(f"❌ Correlation plot failed: {str(e)}")
            return go.Figure()
    
    def create_time_series(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """Create time series plot for temperature and energy demand."""
        try:
            if df.empty:
                st.warning("⚠️ No data for time series")
                return go.Figure()
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set1
            
            for idx, city in enumerate(selected_cities):
                if city not in df['city'].unique():
                    continue
                    
                city_df = df[df['city'] == city]
                if city_df.empty:
                    continue
                
                color = colors[idx % len(colors)]
                
                # Temperature line
                fig.add_trace(go.Scatter(
                    x=city_df['date'],
                    y=city_df['temperature_avg'],
                    name=f'{city} Temperature',
                    line=dict(color=color, width=2),
                    yaxis='y1',
                    hovertemplate=f'<b>{city}</b><br>Date: %{{x}}<br>Temperature: %{{y:.1f}}°F<extra></extra>'
                ))
                
                # Energy line (dashed)
                fig.add_trace(go.Scatter(
                    x=city_df['date'],
                    y=city_df['energy_demand'],
                    name=f'{city} Energy',
                    line=dict(color=color, dash='dot', width=2),
                    yaxis='y2',
                    hovertemplate=f'<b>{city}</b><br>Date: %{{x}}<br>Energy: %{{y:,.0f}} MWh<extra></extra>'
                ))
            
            # Highlight weekends
            weekends = df[df['is_weekend'] == True]['date'].unique()
            for weekend in weekends:
                fig.add_vrect(
                    x0=weekend, x1=weekend + timedelta(days=1),
                    fillcolor="rgba(0, 100, 255, 0.1)",
                    layer="below", line_width=0
                )
            
            fig.update_layout(
                title=dict(
                    text='📈 Temperature and Energy Demand Timeline',
                    x=0.5,
                    font=dict(size=18, color='#1f77b4')
                ),
                xaxis_title='📅 Date',
                yaxis=dict(title='🌡️ Temperature (°F)', color='#d62728', side='left'),
                yaxis2=dict(title='⚡ Energy Demand (MWh)', color='#ff7f0e', overlaying='y', side='right'),
                legend=dict(x=0.01, y=1.05, orientation='h'),
                height=500,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0.95)',
                plot_bgcolor='rgba(0,0,0,0.5)'
            )
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create time series: {str(e)}")
            st.error(f"❌ Time series failed: {str(e)}")
            return go.Figure()
    
    def create_geographic_map(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive map visualization."""
        try:
            if df.empty:
                st.warning("⚠️ No data for geographic map")
                return go.Figure()
            
            # Calculate usage levels
            usage_levels = self.calculate_usage_levels(
                df, 
                selected_cities=self.selected_cities,
                lookback_days=self.lookback_days,
                recent_days=self.recent_days
            )
            
            if not usage_levels:
                st.warning("⚠️ No usage level data available for mapping")
                return go.Figure()
            
            # Calculate last day's changes if enabled
            last_day_changes = {}
            if self.show_last_day_change:
                last_day_changes = self.calculate_last_day_change(df)
            
            # Create map data
            map_data = []
            for city_name, city_data in usage_levels.items():
                change_data = last_day_changes.get(city_name, {})
                change_text = ""
                if change_data:
                    change_pct = change_data['pct_change']
                    change_icon = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                    days_gap = f" ({change_data['days_between']}d gap)" if change_data['days_between'] > 1 else ""
                    change_text = f"<br>{change_icon} Last Change: {change_pct:+.1f}% ({change_data['latest_usage']:,.0f} MWh){days_gap}<br>📅 From {change_data['previous_date']} to {change_data['latest_date']}"
                
                map_data.append({
                    'city': city_data['city_name'],
                    'state': city_data['state'],
                    'lat': city_data['lat'],
                    'lon': city_data['lon'],
                    'current_usage': city_data['current_usage'],
                    'baseline_median': city_data['baseline_median'],
                    'status': city_data['status'],
                    'color': city_data['color'],
                    'status_description': city_data['status_description'],
                    'last_updated': city_data['last_updated'],
                    'change_text': change_text
                })
            
            map_df = pd.DataFrame(map_data)
            
            # Create the map
            fig = go.Figure()
            
            fig.add_trace(go.Scattergeo(
                lon=map_df['lon'],
                lat=map_df['lat'],
                text=map_df['city'],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=map_df['color'],
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                textfont=dict(size=9, color='black'),
                textposition='bottom center',
                hoverinfo='text',
                hovertext=map_df.apply(
                    lambda x: (
                        f"<b>{x['city']}, {x['state']}</b><br>"
                        f"📊 Status: <b>{x['status'].upper()}</b><br>"
                        f"⚡ Current: {x['current_usage']:,.1f} MWh<br>"
                        f"📈 Baseline: {x['baseline_median']:,.1f} MWh<br>"
                        f"🔄 {x['status_description']}<br>"
                        f"📅 Updated: {x['last_updated']}"
                        f"{x['change_text']}"
                    ),
                    axis=1
                ),
                name='Energy Status'
            ))
            
            # Summary stats
            usage_summary = self.get_usage_summary(usage_levels)
            
            fig.update_layout(
                title=dict(
                    text=f"🗺️ US Energy Usage Status Map",
                    x=0.5,
                    font=dict(size=18, color='#1f77b4')
                ),
                geo=dict(
                    scope='usa',
                    projection_type='albers usa',
                    showland=True,
                    landcolor='lightgray',
                    coastlinecolor="white",
                    showlakes=True,
                    lakecolor='lightblue'
                ),
                margin=dict(l=0, r=0, t=60, b=0),
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create geographic map: {str(e)}")
            st.error(f"❌ Geographic map failed: {str(e)}")
            return go.Figure()
    
    def check_data_freshness(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check data freshness and flag if we're getting old data."""
        try:
            if df.empty:
                return {
                    'is_fresh': False,
                    'latest_date': None,
                    'days_old': None,
                    'status': 'No data available',
                    'warning_level': 'critical'
                }
            
        
            
            # Get the most recent date in the dataset
            latest_date = df['date'].max()
            if isinstance(latest_date, pd.Timestamp):
                latest_date = latest_date.date()
            current_date = datetime.now().date()

            # Calculate how many days old the data is
            days_old = (current_date - latest_date).days




            # Define freshness thresholds
            if days_old <= 1:
                status = 'Fresh - data is current'
                warning_level = 'good'
                is_fresh = True
            elif days_old <= 3:
                status = 'Slightly stale - data is 2-3 days old'
                warning_level = 'warning'
                is_fresh = True
            elif days_old <= 7:
                status = 'Stale - data is up to a week old'
                warning_level = 'warning'
                is_fresh = False
            else:
                status = 'Very stale - data is over a week old'
                warning_level = 'critical'
                is_fresh = False
            
            return {
                'is_fresh': is_fresh,
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'days_old': days_old,
                'status': status,
                'warning_level': warning_level
            }
            
        except Exception as e:
            logger.error(f"Failed to check data freshness: {str(e)}")
            return {
                'is_fresh': False,
                'latest_date': None,
                'days_old': None,
                'status': f'Error checking freshness: {str(e)}',
                'warning_level': 'critical'
            }
    
    def _show_summary_metrics(self, df: pd.DataFrame):
        """Show key summary metrics at the top."""
        if df.empty:
            return
            
        # Calculate basic stats
        total_cities = len(df['city'].unique())
        date_range_days = (df['date'].max() - df['date'].min()).days
        avg_energy = df['energy_demand'].mean()
        avg_temp = df['temperature_avg'].mean() if 'temperature_avg' in df.columns else 0
        
        # Show metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏙️ Cities Analyzed",
                value=total_cities,
                help="Number of cities in current analysis"
            )
        
        with col2:
            st.metric(
                label="📅 Data Period",
                value=f"{date_range_days} days",
                help="Number of days in analysis period"
            )
        
        with col3:
            st.metric(
                label="⚡ Avg Energy",
                value=f"{avg_energy:,.0f} MWh",
                help="Average energy demand across all cities"
            )
        
        with col4:
            st.metric(
                label="🌡️ Avg Temperature",
                value=f"{avg_temp:.1f}°F",
                help="Average temperature across all cities"
            )
        
        # Show last day changes if enabled
        if self.show_last_day_change:
            last_day_changes = self.calculate_last_day_change(df)
            if last_day_changes:
                st.markdown("### 📈 Recent Energy Changes")
                
                # Show top 4 changes
                # Show top 4 changes
                sorted_changes = sorted(
                    last_day_changes.items(), 
                    key=lambda x: abs(x[1]['pct_change']), 
                    reverse=True
                )

                change_cols = st.columns(min(5, len(sorted_changes)))
                for idx, (city, change_data) in enumerate(sorted_changes[:5]):
                    with change_cols[idx]:
                        pct_change = change_data['pct_change']
                        
                        days_gap_text = f" ({change_data['days_between']}d)" if change_data['days_between'] > 1 else ""
                        
                        # Try without the + sign and see if that helps
                        delta_text = f"{pct_change:.1f}%" if pct_change >= 0 else f"{pct_change:.1f}%"
                        
                        st.metric(
                            f"{city}{days_gap_text}",
                            f"{change_data['latest_usage']:,.0f} MWh",
                            delta=delta_text,
                            delta_color="inverse",
                            help=f"Change from {change_data['previous_date']} to {change_data['latest_date']}"
                        )
                        
                        



                # Expandable section for all changes
                if len(last_day_changes) > 4:
                    with st.expander(f"📋 View All {len(last_day_changes)} Cities' Changes"):
                        change_df = pd.DataFrame([
                            {
                                'City': city,
                                'Latest (MWh)': f"{data['latest_usage']:,.0f}",
                                'Previous (MWh)': f"{data['previous_usage']:,.0f}",
                                'Change (%)': f"{data['pct_change']:+.1f}%",
                                'Latest Date': data['latest_date'],
                                'Previous Date': data['previous_date']
                            }
                            for city, data in sorted_changes
                        ])
                        st.dataframe(change_df, use_container_width=True, hide_index=True)
    
    def _show_data_quality(self, df: pd.DataFrame):
        """Display enhanced data quality report including data freshness."""
        if not self.show_quality or df.empty:
            return
        
        with st.expander("🔍 Data Quality Report"):
            quality_report = self.generate_data_quality_report(df)
            summary = quality_report['summary']
            
            # Check data freshness
            freshness_info = self.check_data_freshness(df)
            
            # Quality metrics - now including freshness
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Records", summary['total_rows'])
            with col2:
                st.metric("❌ Missing Values", summary['missing_values'])
            with col3:
                st.metric("⚠️ Outliers", summary['outliers'])
            with col4:
                # Data freshness metric with color coding
                if freshness_info['warning_level'] == 'good':
                    st.metric("🟢 Data Freshness", "Fresh", delta=f"{freshness_info['days_old']} days old")
                elif freshness_info['warning_level'] == 'warning':
                    st.metric("🟡 Data Freshness", "Stale", delta=f"{freshness_info['days_old']} days old")
                else:
                    st.metric("🔴 Data Freshness", "Very Stale", delta=f"{freshness_info['days_old']} days old")
            
            # Freshness details
            st.markdown("#### 📅 Data Freshness Analysis")
            
            # Show freshness status with appropriate styling
            if freshness_info['warning_level'] == 'good':
                st.success(f"✅ **{freshness_info['status']}**")
            elif freshness_info['warning_level'] == 'warning':
                st.warning(f"⚠️ **{freshness_info['status']}**")
            else:
                st.error(f"🚨 **{freshness_info['status']}**")
            
            # Additional freshness details
            if freshness_info['latest_date']:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"**Latest Data:** {freshness_info['latest_date']}")
                with col_b:
                    st.info(f"**Current Date:** {datetime.now().strftime('%Y-%m-%d')}")
            
            # Issues found
            st.markdown("#### 🚨 Quality Issues")
            all_issues = []
            
            # Add freshness issues
            if not freshness_info['is_fresh']:
                if freshness_info['days_old'] and freshness_info['days_old'] > 7:
                    all_issues.append(f"⏱️ Data is {freshness_info['days_old']} days old - consider refreshing data sources")
                elif freshness_info['days_old'] and freshness_info['days_old'] > 3:
                    all_issues.append(f"⏱️ Data is {freshness_info['days_old']} days old - may impact real-time insights")
            
            # Add existing quality issues
            if quality_report['issues']:
                all_issues.extend(quality_report['issues'])
            
            if all_issues:
                for issue in all_issues:
                    st.write(f"• {issue}")
            else:
                st.success("✅ No data quality issues detected!")
            
            # Recommendations based on freshness
            if freshness_info['days_old'] and freshness_info['days_old'] > 1:
                st.markdown("#### 💡Insight ")
                if freshness_info['days_old'] > 7:
                    st.write("• 📞 **Action:** Check data pipeline and API connections")
                elif freshness_info['days_old'] > 3:
                    st.write("• 📈 **Impact:** Recent trends may not be fully captured")
                else:
                    st.write("• 🔄 **Stale reason:** Data is 2 days delayed to wait for data availability before fetching")
    
    def display(self):
        """Main dashboard display method with improved layout."""
        # Check if cities are selected
        if not self.selected_cities:
            st.error("⚠️ Please select at least one city from the sidebar to proceed with the analysis.")
            return
        
        # Load and filter data
        with st.spinner("🔄 Loading data..."):
            data = self._load_data(self.date_range)
            filtered_data = self._filter_data(data)
        
        if filtered_data.empty:
            st.warning("⚠️ No data available for the selected filters. Please adjust your selection in the sidebar.")
            return
        
        # Show summary metrics
        st.markdown("## 📊 Analysis Summary")
        self._show_summary_metrics(filtered_data)
        
        # Show data quality if enabled
        self._show_data_quality(filtered_data)
        
        # Show historical data table
        self._show_historical_data_table(filtered_data)
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographic View", "📈 Time Series", "📊 Correlations", "🔥 Usage Patterns"])
        
        with tab1:
            st.markdown("### 🗺️ Geographic Energy Usage Overview")
            map_fig = self.create_geographic_map(filtered_data)
            st.plotly_chart(map_fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📈 Temperature & Energy Timeline")
            ts_fig = self.create_time_series(filtered_data, self.selected_cities)
            st.plotly_chart(ts_fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 📊 Temperature vs Energy Correlation")
            corr_fig = self.create_correlation_plot(filtered_data, self.selected_cities)
            st.plotly_chart(corr_fig, use_container_width=True)
        
        with tab4:
            st.markdown("### 🔥 Energy Usage Patterns by Temperature & Day")
            st.info(f"📋 Displaying heatmap for {len(self.selected_cities)} selected cities with day-of-week labels")
            heatmap_fig = self.create_heatmap(filtered_data, self.selected_cities)
            st.plotly_chart(heatmap_fig, use_container_width=True)

def check_pipeline_exists():
    """Check if pipeline.py exists and return its path."""
    from pathlib import Path
    import os
    
    # Get the actual current working directory
    current_dir = Path.cwd()
    
    # List of possible locations for pipeline.py
    possible_paths = [
        current_dir / "pipeline.py",  # Current directory (src)
        current_dir / "src" / "pipeline.py",  # In case we're in parent dir
        current_dir.parent / "src" / "pipeline.py",  # Parent/src structure
        Path(__file__).parent / "pipeline.py" if '__file__' in globals() else current_dir / "pipeline.py",
    ]
    
    # Debug: show where we're looking
    print(f"Current working directory: {current_dir}")
    print(f"Looking for pipeline.py in these locations:")
    for i, path in enumerate(possible_paths, 1):
        exists = path.exists()
        print(f"  {i}. {path} {'✓' if exists else '✗'}")
        if exists:
            return path, True
    
    return possible_paths[0], False  # Return first path as default, but indicate not found

def should_run_pipeline(config):
    """Check if pipeline should run based on data freshness (3+ days)."""
    try:
        from pathlib import Path
        import os
        from datetime import datetime
        
        # First check if pipeline.py exists
        pipeline_path, exists = check_pipeline_exists()
        if not exists:
            st.warning(f"⚠️ pipeline.py not found at expected location: {pipeline_path}")
            st.info("Please make sure pipeline.py is in the same directory as your main script.")
            return False  # Don't try to run if file doesn't exist
        
        data_dir = Path(config.data_paths.get('processed', 'data/processed'))
        if not data_dir.exists():
            return True
            
        # Check age of newest file
        newest_file = max(data_dir.glob('*.csv'), key=os.path.getctime, default=None)
        if not newest_file:
            return True
            
        # Calculate file age in days
        file_age_days = (datetime.now().timestamp() - os.path.getctime(newest_file)) / 86400  # 86400 seconds in a day
        return file_age_days >= 3  # Run if data is 3 or more days old
        
    except Exception as e:
        logger.warning(f"Pipeline check failed: {e}")
        return True  # Run pipeline if check fails

def run_pipeline_with_progress():
    """Run pipeline with animated progress bar and engaging messages."""
    import subprocess
    import sys
    import threading
    import time
    import random
    from pathlib import Path
    
    # Create progress bar and status placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Progress tracking
    progress_value = 0
    pipeline_complete = False
    pipeline_result = None
    message_index = 0
    
    # Engaging messages for different stages
    progress_messages = [
        # 0-10%: Starting up
        "🚀 Initializing data pipeline...",
        "📡 Connecting to data sources...",
        "🔍 Scanning for available datasets...",
        
        # 10-30%: Data fetching
        "📊 Fetching energy consumption data...",
        "🌡️ Downloading weather information...",
        "📈 Retrieving historical trends...",
        "🔄 Synchronizing data streams...",
        
        # 30-50%: Processing
        "⚙️ Processing raw data files...",
        "🧹 Cleaning and validating datasets...",
        "🔗 Linking weather and energy data...",
        "📏 Normalizing data formats...",
        
        # 50-70%: Analysis
        "🔬 Analyzing consumption patterns...",
        "📊 Computing statistical correlations...",
        "🎯 Identifying peak usage periods...",
        "🌟 Detecting seasonal trends...",
        
        # 70-90%: Finalization
        "💾 Saving processed datasets...",
        "📋 Generating summary reports...",
        "🔧 Optimizing data structures...",
        "✨ Preparing visualizations...",
        
        # 90-95%: Final steps
        "🎨 Finalizing dashboard components...",
        "🔍 Running quality checks...",
        "📦 Packaging results..."
    ]
    
    def run_pipeline():
        nonlocal pipeline_complete, pipeline_result
        try:
            # Get the pipeline path
            pipeline_path, exists = check_pipeline_exists()
            
            if not exists:
                raise FileNotFoundError(f"pipeline.py not found at any expected location")
            
            print(f"Found pipeline.py at: {pipeline_path}")
            
            # Set working directory to where pipeline.py is located
            working_dir = pipeline_path.parent
            print(f"Running pipeline from directory: {working_dir}")
            
            # Check if config directory exists in the working directory
            config_dir = working_dir / "config"
            if not config_dir.exists():
                # Try to find config directory in parent or other locations
                possible_config_dirs = [
                    working_dir.parent / "config",  # Parent directory
                    working_dir / ".." / "config",  # Relative parent
                    Path.cwd() / "config",  # Current working directory
                ]
                
                for config_path in possible_config_dirs:
                    if config_path.exists():
                        print(f"Found config directory at: {config_path.resolve()}")
                        # Update working directory to parent of config
                        working_dir = config_path.parent
                        break
                else:
                    print(f"Warning: config directory not found. Searched in:")
                    for cp in [config_dir] + possible_config_dirs:
                        print(f"  - {cp}")
            
            # Run pipeline with correct working directory
            pipeline_result = subprocess.run(
                [sys.executable, str(pipeline_path)], 
                capture_output=True, 
                text=True, 
                cwd=str(working_dir)
            )
            pipeline_complete = True
        except Exception as e:
            pipeline_result = e
            pipeline_complete = True
    
    # Start pipeline in background thread
    pipeline_thread = threading.Thread(target=run_pipeline)
    pipeline_thread.start()
    
    # Animate progress bar - much slower for 2-3 minute process
    start_time = time.time()
    while not pipeline_complete:
        elapsed_time = time.time() - start_time
        
        # Slower, more realistic progress calculation
        # Assume 150 seconds (2.5 minutes) total time
        expected_duration = 150  # seconds
        time_based_progress = min(int((elapsed_time / expected_duration) * 90), 90)  # Cap at 90%
        
        # Gradual increment with some randomness
        if progress_value < time_based_progress:
            progress_value = min(progress_value + random.uniform(0.5, 1.5), time_based_progress)
        
        # Update progress bar
        progress_bar.progress(int(progress_value))
        
        # Update message based on progress
        message_stage = min(int(progress_value / 4), len(progress_messages) - 1)
        if message_stage != message_index:
            message_index = message_stage
        
        current_message = progress_messages[message_index]
        
        # Add animated dots and time info
        dots = "." * ((int(elapsed_time * 2) % 3) + 1)
        elapsed_str = f"{int(elapsed_time//60)}:{int(elapsed_time%60):02d}"
        
        status_text.text(f"{current_message}{dots} ({int(progress_value)}%) - {elapsed_str}")
        
        time.sleep(0.8)  # Slower update interval
    
    # Complete the progress bar
    progress_bar.progress(100)
    status_text.text("🎉 Pipeline completed successfully! (100%)")
    time.sleep(1)  # Brief pause to show completion
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Return pipeline result
    pipeline_thread.join()  # Ensure thread completes
    return pipeline_result

def main():
    """Main entry point for the dashboard."""
    try:
        config = Config.load()
        
        # Check for manual refresh trigger
        if st.session_state.get('manual_refresh', False):
            st.session_state.manual_refresh = False  # Reset the trigger
            st.info("🚀 Manually refreshing data pipeline...")
            
            # Run pipeline with animated progress
            result = run_pipeline_with_progress()
            
            if isinstance(result, Exception):
                st.error(f"❌ Pipeline failed: {str(result)}")
            elif result.returncode == 0:
                st.success("✅ Data pipeline completed successfully!")
                st.balloons()  # Celebration animation!
            else:
                st.error(f"❌ Pipeline failed: {result.stderr}")
        
        # Check if pipeline should run automatically (only if data is 3+ days old)
        elif should_run_pipeline(config):
            st.info("🚀 Auto-updating data pipeline (data is 3+ days old)...")
            
            # Run pipeline with animated progress
            result = run_pipeline_with_progress()
            
            if isinstance(result, Exception):
                st.error(f"❌ Pipeline failed: {str(result)}")
            elif result.returncode == 0:
                st.success("✅ Data pipeline completed successfully!")
                st.balloons()  # Celebration animation!
            else:
                st.error(f"❌ Pipeline failed: {result.stderr}")
        else:
            # Show data status and manual refresh option
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("📊 Using existing data (updated within last 3 days)")
            with col2:
                if st.button("🔄 Refresh Now", key="manual_refresh_btn", help="Manually refresh data pipeline"):
                    st.session_state.manual_refresh = True
                    st.rerun()
            
        dashboard = Dashboard(config)
        dashboard.display()
        
    except Exception as e:
        st.error("❌ Failed to initialize dashboard. Please check the logs.")
        logger.critical(f"Dashboard failed: {str(e)}")

if __name__ == "__main__":
    main()