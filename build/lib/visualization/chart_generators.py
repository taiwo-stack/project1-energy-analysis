"""Chart generation and visualization utilities for the dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from loguru import logger


class ChartGenerator:
    """Handles all chart and visualization generation for the dashboard."""
    
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
            rows, cols = self._calculate_subplot_layout(len(cities_to_plot))
            
            # Limit cities for display if too many
            if len(cities_to_plot) > 16:
                cities_to_plot = cities_to_plot[:16]
                st.info(f"ℹ️ Showing heatmap for first 16 cities: {', '.join(cities_to_plot)}")
            
            # Create subplots
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"<b>{city}</b>" for city in cities_to_plot],
                vertical_spacing=0.12,
                horizontal_spacing=0.08
            )
            
            # Generate heatmaps for each city
            all_values = []
            for idx, city in enumerate(cities_to_plot):
                row = (idx // cols) + 1
                col = (idx % cols) + 1
                
                city_heatmap, valid_values = self._create_city_heatmap(df, city, temp_order, days_order)
                all_values.extend(valid_values)
                
                # Add heatmap trace
                self._add_heatmap_trace(fig, city_heatmap, city, idx, row, col)
            
            # Set consistent color range and update layout
            self._finalize_heatmap(fig, all_values, cities_to_plot, rows, cols)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create heatmap: {str(e)}")
            st.error(f"❌ Heatmap failed: {str(e)}")
            return go.Figure()
    
    def _calculate_subplot_layout(self, num_cities: int) -> tuple:
        """Calculate optimal subplot layout for given number of cities."""
        if num_cities == 1:
            return 1, 1
        elif num_cities == 2:
            return 1, 2
        elif num_cities <= 4:
            return 2, 2
        elif num_cities <= 6:
            return 2, 3
        elif num_cities <= 9:
            return 3, 3
        elif num_cities <= 12:
            return 3, 4
        else:
            return 4, 4
    
    def _create_city_heatmap(self, df: pd.DataFrame, city: str, temp_order: List[str], days_order: List[str]) -> tuple:
        """Create heatmap data for a specific city."""
        city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand', 'temp_range', 'day_of_week'])
        
        if len(city_df) < 5:
            # Return empty heatmap for insufficient data
            empty_heatmap = pd.DataFrame(
                data=np.full((len(temp_order), len(days_order)), np.nan),
                index=temp_order,
                columns=days_order
            )
            return empty_heatmap, []
        
        try:
            city_heatmap = city_df.groupby(['temp_range', 'day_of_week'], observed=True)['energy_demand'].mean().unstack(fill_value=np.nan)
            city_heatmap = city_heatmap.reindex(index=temp_order, columns=days_order, fill_value=np.nan)
        except Exception as group_error:
            logger.warning(f"Groupby failed for {city}: {str(group_error)}")
            city_heatmap = pd.DataFrame(
                data=np.full((len(temp_order), len(days_order)), np.nan),
                index=temp_order,
                columns=days_order
            )
        
        # Get valid values for color scaling
        valid_values = city_heatmap.values[~np.isnan(city_heatmap.values)]
        
        return city_heatmap, list(valid_values)
    
    def _add_heatmap_trace(self, fig: go.Figure, city_heatmap: pd.DataFrame, city: str, idx: int, row: int, col: int):
        """Add heatmap trace to the figure."""
        # Prepare text for display
        text_values = np.where(
            np.isnan(city_heatmap.values), 
            '', 
            np.vectorize(lambda x: str(int(round(x))) if not np.isnan(x) else '')(city_heatmap.values)
        )
        
        fig.add_trace(
            go.Heatmap(
                z=city_heatmap.values,
                x=city_heatmap.columns,
                y=city_heatmap.index,
                colorscale='RdYlBu_r',
                text=text_values,
                texttemplate='%{text}',
                textfont=dict(size=8),
                showscale=(idx == 0),
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
    
    def _finalize_heatmap(self, fig: go.Figure, all_values: List[float], cities_to_plot: List[str], rows: int, cols: int):
        """Finalize heatmap layout and styling."""
        # Set consistent color range
        if all_values:
            zmin, zmax = min(all_values), max(all_values)
            for trace in fig.data:
                if hasattr(trace, 'zmin'):
                    trace.update(zmin=zmin, zmax=zmax)
        
        # Calculate dynamic height
        base_height = 250
        fig_height = max(base_height * rows, 400)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'🔥 Energy Usage Patterns by Temperature & Day ({len(cities_to_plot)} Cities)',
                x=0.5,
                font=dict(size=18, color='#1f77b4')
            ),
            height=fig_height,
            margin=dict(l=50, r=100, t=60, b=40),
            font=dict(size=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update axes
        for i in range(len(cities_to_plot)):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.update_xaxes(
                showticklabels=True,
                tickangle=45,
                tickfont=dict(size=8),
                row=row, col=col
            )
            
            if col == 1:
                fig.update_yaxes(
                    showticklabels=True,
                    tickfont=dict(size=8),
                    row=row, col=col
                )
            else:
                fig.update_yaxes(showticklabels=False, row=row, col=col)
    
    def create_correlation_plot(self, df: pd.DataFrame, selected_cities: List[str], 
                              calculate_regression_func: Callable, calculate_correlations_func: Callable,
                              show_correlations: bool) -> go.Figure:
        """Create enhanced scatter plot with regression line."""
        try:
            if df.empty:
                st.warning("⚠️ No data for correlation plot")
                return go.Figure()
            
            df = df.copy()
            if 'temperature_avg' not in df.columns:
                df['temperature_avg'] = df.get('temperature_max', df.get('temperature_min', np.nan))
            
            fig = go.Figure()
            
            # Get regression and correlation data
            regression_stats = calculate_regression_func(df, selected_cities)
            correlations = calculate_correlations_func(df, selected_cities)
            
            colors = px.colors.qualitative.Set1
            
            # Add traces for each city
            for idx, city in enumerate(selected_cities):
                if city not in df['city'].unique():
                    continue
                    
                city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand'])
                if city_df.empty:
                    continue
                
                color = colors[idx % len(colors)]
                
                # Add scatter points
                self._add_scatter_trace(fig, city_df, city, color)
                
                # Add regression line
                self._add_regression_line(fig, city_df, city, color, regression_stats, correlations)
            
            # Add correlation annotation if enabled
            if show_correlations and correlations:
                self._add_correlation_annotation(fig, correlations)
            
            # Update layout
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
    
    def _add_scatter_trace(self, fig: go.Figure, city_df: pd.DataFrame, city: str, color: str):
        """Add scatter trace for a city."""
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
    
    def _add_regression_line(self, fig: go.Figure, city_df: pd.DataFrame, city: str, color: str, 
                           regression_stats: Dict, correlations: Dict):
        """Add regression line for a city."""
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
    
    def _add_correlation_annotation(self, fig: go.Figure, correlations: Dict):
        """Add correlation annotation to the plot."""
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
    
    def create_time_series(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """Create time series plot for temperature and energy demand."""
        try:
            if df.empty:
                st.warning("⚠️ No data for time series")
                return go.Figure()
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set1
            
            # Add temperature and energy lines for each city
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
                    y=city_df['temperature_avg'] if 'temperature_avg' in city_df.columns else city_df.get('temperature_max', 0),
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
            self._highlight_weekends(fig, df)
            
            # Update layout
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
    
    def _highlight_weekends(self, fig: go.Figure, df: pd.DataFrame):
        """Add weekend highlighting to time series plot."""
        if 'is_weekend' in df.columns:
            weekends = df[df['is_weekend'] == True]['date'].unique()
            for weekend in weekends:
                fig.add_vrect(
                    x0=weekend, x1=weekend + timedelta(days=1),
                    fillcolor="rgba(0, 100, 255, 0.1)",
                    layer="below", line_width=0
                )
    
    def create_geographic_map(self, df: pd.DataFrame, calculate_usage_levels_func: Callable,
                            calculate_last_day_change_func: Optional[Callable], selected_cities: List[str],
                            lookback_days: int, recent_days: int) -> go.Figure:
        """Create interactive map visualization."""
        try:
            if df.empty:
                st.warning("⚠️ No data for geographic map")
                return go.Figure()
            
            # Calculate usage levels
            usage_levels = calculate_usage_levels_func(
                df, 
                selected_cities=selected_cities,
                lookback_days=lookback_days,
                recent_days=recent_days
            )
            
            if not usage_levels:
                st.warning("⚠️ No usage level data available for mapping")
                return go.Figure()
            
            # Calculate last day's changes if function provided
            last_day_changes = {}
            if calculate_last_day_change_func:
                last_day_changes = calculate_last_day_change_func(df)
            
            # Create map data
            map_data = self._prepare_map_data(usage_levels, last_day_changes)
            map_df = pd.DataFrame(map_data)
            
            # Create the map
            fig = self._create_map_figure(map_df)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create geographic map: {str(e)}")
            st.error(f"❌ Geographic map failed: {str(e)}")
            return go.Figure()
    
    def _prepare_map_data(self, usage_levels: Dict, last_day_changes: Dict) -> List[Dict]:
        """Prepare data for map visualization."""
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
        
        return map_data
    
    def _create_map_figure(self, map_df: pd.DataFrame) -> go.Figure:
        """Create the map figure with markers."""
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
    
    def create_energy_distribution_chart(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """Create energy distribution box plot or violin plot."""
        try:
            if df.empty:
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
                
                fig.add_trace(go.Box(
                    y=city_df['energy_demand'],
                    name=city,
                    marker_color=color,
                    boxpoints='outliers',
                    hovertemplate=f'<b>{city}</b><br>Energy: %{{y:,.0f}} MWh<extra></extra>'
                ))
            
            fig.update_layout(
                title=dict(
                    text='📊 Energy Demand Distribution by City',
                    x=0.5,
                    font=dict(size=18, color='#1f77b4')
                ),
                yaxis_title='⚡ Energy Demand (MWh)',
                xaxis_title='🏙️ Cities',
                height=500,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.5)'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create distribution chart: {str(e)}")
            return go.Figure()
    
    def create_seasonal_trend_chart(self, df: pd.DataFrame, selected_cities: List[str]) -> go.Figure:
        """Create seasonal trend analysis chart."""
        try:
            if df.empty or 'date' not in df.columns:
                return go.Figure()
            
            # Add month column for seasonal analysis
            df = df.copy()
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['month_name'] = pd.to_datetime(df['date']).dt.strftime('%B')
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set1
            
            for idx, city in enumerate(selected_cities):
                if city not in df['city'].unique():
                    continue
                
                city_df = df[df['city'] == city]
                if city_df.empty:
                    continue
                
                # Calculate monthly averages
                monthly_avg = city_df.groupby('month_name')['energy_demand'].mean().reset_index()
                
                color = colors[idx % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=monthly_avg['month_name'],
                    y=monthly_avg['energy_demand'],
                    mode='lines+markers',
                    name=city,
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{city}</b><br>Month: %{{x}}<br>Avg Energy: %{{y:,.0f}} MWh<extra></extra>'
                ))
            
            fig.update_layout(
                title=dict(
                    text='📈 Seasonal Energy Consumption Trends',
                    x=0.5,
                    font=dict(size=18, color='#1f77b4')
                ),
                xaxis_title='📅 Month',
                yaxis_title='⚡ Average Energy Demand (MWh)',
                height=500,
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.5)'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create seasonal trend chart: {str(e)}")
            return go.Figure()