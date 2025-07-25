"""Interactive Streamlit dashboard for weather and energy visualizations."""

import streamlit as st
from streamlit.runtime.caching import cache_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from config import Config
from analysis import Analyzer
from loguru import logger
from pathlib import Path
import glob
import json
import numpy as np

class Dashboard(Analyzer):
    """Interactive dashboard extending Analyzer for visualizations."""
    
    def __init__(self, config: Config):
        """Initialize Dashboard with configuration."""
        super().__init__(config)
        self._setup_page()
        self._setup_sidebar()
        self._setup_logging()
    
    def _setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="US Weather and Energy Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("US Weather and Energy Analysis Dashboard")
    
    def _setup_sidebar(self):
        """Configure sidebar filters and controls."""
        with st.sidebar:
            st.header("Filters and Controls")
            
            # Date range selector with 90-day limit note
            default_end = datetime.now().date() - timedelta(days=self.buffer_days)
            default_start = default_end - timedelta(days=self.max_fetch_days)
            st.write(f"**Note**: Data limited to last {self.max_fetch_days} days, ending {self.buffer_days} days ago.")
            self.date_range = st.date_input(
                "Date Range",
                value=(default_start, default_end),
                min_value=default_start,
                max_value=default_end
            )
            
            # City selector
            self.cities = [city.name for city in self.config.cities] + ['All Cities']
            self.selected_cities = st.multiselect(
                "Cities",
                options=self.cities,
                default=self.cities[:-1]
            )
            
            # Time series city selector
            self.time_series_city = st.selectbox(
                "Time Series City",
                options=self.cities,
                index=0
            )
            
            # Display options
            st.header("Display Options")
            self.show_quality = st.checkbox("Show Data Quality Report", value=True)
    
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
    
    @cache_data(ttl=3600, show_spinner="Loading data...")
    def _load_data(_self, date_range: Tuple[datetime.date, datetime.date]) -> pd.DataFrame:
        """Load and cache processed data using Analyzer."""
        return _self.load_data(date_range)
    
    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data based on user selections.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Filtered DataFrame
        """
        if 'All Cities' not in self.selected_cities and self.selected_cities:
            df = df[df['city'].isin(self.selected_cities)]
        if self.time_series_city != 'All Cities':
            df = df[df['city'] == self.time_series_city]
        return df
    
    def create_heatmap(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """
        Create heatmap of energy demand by temperature range and day of week.
        
        Args:
            df: DataFrame with data
            selected_cities: Cities to include
        
        Returns:
            Plotly Figure
        """
        try:
            if df.empty:
                st.warning("⚠️ No data for heatmap")
                return go.Figure()
            
            df = df.copy()
            if 'temperature_avg' not in df.columns:
                df['temperature_avg'] = df.get('temperature_max', df.get('temperature_min', np.nan))
            
            # Define temperature bins for heatmap
            bins = [-float('inf'), 50, 60, 70, 80, 90, float('inf')]
            labels = ['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
            df['temp_range'] = pd.cut(df['temperature_avg'], bins=bins, labels=labels, include_lowest=True)
            
            fig = go.Figure()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            temp_order = labels
            
            for city in df['city'].unique():
                if city not in selected_cities and 'All Cities' not in selected_cities:
                    continue
                city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand', 'temp_range', 'day_of_week'])
                if len(city_df) < 2:
                    st.warning(f"⚠️ Skipping heatmap for {city}: insufficient data")
                    continue
                
                city_heatmap = city_df.groupby(['temp_range', 'day_of_week'], observed=False)['energy_demand'].mean().unstack()
                city_heatmap = city_heatmap.reindex(index=temp_order, columns=days_order, fill_value=np.nan)
                
                fig.add_trace(go.Heatmap(
                    z=city_heatmap.values,
                    x=city_heatmap.columns,
                    y=city_heatmap.index,
                    colorscale='RdBu',
                    text=city_heatmap.values.round(1),
                    texttemplate='%{text}',
                    name=city,
                    colorbar=dict(title='Energy Demand (MWh)'),
                    hovertemplate=(
                        f'<b>{city}</b><br>'
                        'Day: %{x}<br>'
                        'Temp Range: %{y}<br>'
                        'Energy: %{z:,.0f} MWh<br>'
                        '<extra></extra>'
                    )
                ))
            
            if not fig.data:
                st.warning("⚠️ No valid heatmap data")
                return go.Figure()
            
            fig.update_layout(
                title='📊 Average Energy Usage by Temperature and Day of Week',
                xaxis_title='Day of Week',
                yaxis_title='Temperature Range',
                height=600,
                margin=dict(l=80, r=80, t=80, b=80),
                showlegend=True
            )
            return fig
            
        except Exception as e:
            logger.error("Failed to create heatmap: {}", str(e))
            st.error(f"❌ Heatmap failed: {str(e)}")
            return go.Figure()
    
    def create_correlation_plot(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """
        Create scatter plot with regression line for temperature vs. energy demand.
        
        Args:
            df: DataFrame with data
            selected_cities: Cities to include
        
        Returns:
            Plotly Figure
        """
        try:
            if df.empty:
                st.warning("⚠️ No data for correlation plot")
                return go.Figure()
            
            df = df.copy()
            if 'temperature_avg' not in df.columns:
                df['temperature_avg'] = df.get('temperature_max', df.get('temperature_min', np.nan))
            
            fig = go.Figure()
            regression_stats = self.calculate_regression(df, selected_cities)
            
            for city in df['city'].unique():
                if city not in selected_cities and 'All Cities' not in selected_cities:
                    continue
                city_df = df[df['city'] == city]
                if city_df.empty:
                    continue
                
                fig.add_trace(go.Scatter(
                    x=city_df['temperature_avg'],
                    y=city_df['energy_demand'],
                    mode='markers',
                    name=city,
                    marker=dict(color=self.config.city_colors.get(city, 'blue')),
                    text=city_df['date'].dt.strftime('%Y-%m-%d'),
                    hovertemplate=(
                        f'<b>{city}</b><br>'
                        'Date: %{text}<br>'
                        'Temperature: %{x:.1f}°F<br>'
                        'Energy: %{y:,.0f} MWh<br>'
                        '<extra></extra>'
                    )
                ))
                
                stats = regression_stats.get(city, {})
                if not np.isnan(stats.get('slope')):
                    x_range = [city_df['temperature_avg'].min(), city_df['temperature_avg'].max()]
                    y_pred = [stats['slope'] * x + stats['intercept'] for x in x_range]
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name=f'{city} Regression (R²={stats["r_squared"]:.3f})',
                        line=dict(color=self.config.city_colors.get(city, 'blue'), dash='dash')
                    ))
            
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref='paper',
                yref='paper',
                text=(
                    f"Correlations:<br>" +
                    "<br>".join([f"{city}: {corr:.3f}" for city, corr in self.calculate_correlations(df, selected_cities).items()])
                ),
                showarrow=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
            
            fig.update_layout(
                title='📊 Temperature vs. Energy Demand',
                xaxis_title='🌡️ Average Temperature (°F)',
                yaxis_title='⚡ Energy Demand (MWh)',
                height=600,
                showlegend=True
            )
            return fig
            
        except Exception as e:
            logger.error("Failed to create correlation plot: {}", str(e))
            st.error(f"❌ Correlation plot failed: {str(e)}")
            return go.Figure()
    
    def create_time_series(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """
        Create time series plot for temperature and energy demand.
        
        Args:
            df: DataFrame with data
            selected_cities: Cities to include
        
        Returns:
            Plotly Figure
        """
        try:
            if df.empty:
                st.warning("⚠️ No data for time series")
                return go.Figure()
            
            fig = go.Figure()
            cities = selected_cities if 'All Cities' not in selected_cities else df['city'].unique()
            
            for city in cities:
                city_df = df[df['city'] == city]
                if city_df.empty:
                    continue
                
                fig.add_trace(go.Scatter(
                    x=city_df['date'],
                    y=city_df['temperature_avg'],
                    name=f'{city} Temp (°F)',
                    line=dict(color=self.config.city_colors.get(city, 'blue')),
                    yaxis='y1'
                ))
                fig.add_trace(go.Scatter(
                    x=city_df['date'],
                    y=city_df['energy_demand'],
                    name=f'{city} Energy (MWh)',
                    line=dict(color=self.config.city_colors.get(city, 'red'), dash='dot'),
                    yaxis='y2'
                ))
            
            weekends = df[df['is_weekend']]['date'].unique()
            for weekend in weekends:
                fig.add_vrect(
                    x0=weekend,
                    x1=weekend + timedelta(days=1),
                    fillcolor="rgba(0, 0, 255, 0.1)",
                    layer="below",
                    line_width=0
                )
            
            fig.update_layout(
                title='📈 Temperature and Energy Demand Over Time',
                xaxis_title='📅 Date',
                yaxis=dict(title='🌡️ Temperature (°F)', color='blue'),
                yaxis2=dict(title='⚡ Energy Demand (MWh)', color='red', overlaying='y', side='right'),
                legend=dict(x=0.01, y=1.05, orientation='h'),
                height=600,
                hovermode='x unified'
            )
            return fig
            
        except Exception as e:
            logger.error("Failed to create time series: {}", str(e))
            st.error(f"❌ Time series failed: {str(e)}")
            return go.Figure()
    
    def create_geographic_map(self, df: pd.DataFrame) -> go.Figure:
        """
        Create interactive map visualization of current data.
        
        Args:
            df: DataFrame with data
        
        Returns:
            Plotly Figure
        """
        try:
            latest_date = df['date'].max()
            latest_data = df[df['date'] == latest_date]
            
            if latest_data.empty:
                st.warning("⚠️ No data for geographic map")
                return go.Figure()
            
            map_data = []
            for city in self.config.cities:
                if city.name in self.selected_cities or 'All Cities' in self.selected_cities:
                    city_data = latest_data[latest_data['city'] == city.name]
                    if not city_data.empty:
                        row = city_data.iloc[0]
                        prev_day = df[
                            (df['city'] == city.name) & 
                            (df['date'] == latest_date - timedelta(days=1))
                        ]
                        energy_change = 0
                        if not prev_day.empty and pd.notna(row['energy_demand']) and pd.notna(prev_day['energy_demand'].iloc[0]):
                            energy_change = (
                                (row['energy_demand'] - prev_day['energy_demand'].iloc[0]) / 
                                prev_day['energy_demand'].iloc[0] * 100
                            )
                        
                        map_data.append({
                            'city': city.name,
                            'lat': city.lat,
                            'lon': city.lon,
                            'temperature': row.get('temperature_avg'),
                            'energy': row.get('energy_demand'),
                            'energy_change': energy_change
                        })
            
            if not map_data:
                st.warning("⚠️ No valid map data")
                return go.Figure()
            
            map_df = pd.DataFrame(map_data)
            map_df['color'] = pd.cut(
                map_df['energy'],
                bins=5,
                labels=['#00FF00', '#80FF00', '#FFFF00', '#FF8000', '#FF0000']
            )
            
            fig = go.Figure(data=go.Scattergeo(
                lon=map_df['lon'],
                lat=map_df['lat'],
                text=map_df.apply(
                    lambda x: f"<b>{x['city']}</b><br>"
                    f"🌡️ Temp"
                , axis=1),
                marker=dict(
                    size=16,
                    color=map_df['color'],
                    line=dict(width=1, color='black')
                ),
                mode='markers'
            ))
            return fig

        except Exception as e:
            logger.error("Failed to create geographic map: {}", str(e))
            st.error(f"❌ Geographic map failed: {str(e)}")
            return go.Figure()