"""Visualization components for the dashboard."""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List
import streamlit as st
from loguru import logger


class ChartGenerator:
    """Generates various chart types for the dashboard."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
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
                cities_to_plot = cities_to_plot[:16]
                st.info(f"ℹ️ Showing heatmap for first 16 cities: {', '.join(cities_to_plot)}")
            
            # Create subplots with better spacing
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
                
                if len(city_df) < 5:
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
                    continue
                
                # Create heatmap data
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
                
                # Collect values for color scale
                valid_values = city_heatmap.values[~np.isnan(city_heatmap.values)]
                if len(valid_values) > 0:
                    all_values.extend(valid_values)
                
                # Prepare text for display
                text_values = np.where(
                    np.isnan(city_heatmap.values), 
                    '', 
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
            
            # Set consistent color range for all heatmaps
            if all_values:
                zmin, zmax = min(all_values), max(all_values)
                for trace in fig.data:
                    if hasattr(trace, 'zmin'):
                        trace.update(zmin=zmin, zmax=zmax)
            
            # Calculate dynamic height based on number of rows
            base_height = 250
            fig_height = max(base_height * rows, 400)
            
            # Update layout with better styling
            fig.update_layout(
                title=dict(
                    text=f'🔥 Energy Usage Patterns by Avg. Temperature & Day ({num_cities} Cities)',
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
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create heatmap: {str(e)}")
            st.error(f"❌ Heatmap failed: {str(e)}")
            return go.Figure()
    


    def create_correlation_plot(self, df: pd.DataFrame, selected_cities: List[str], show_correlations: bool = True) -> go.Figure:
        """Create enhanced scatter plot with regression line and comprehensive statistics."""
        try:
            if df.empty:
                st.warning("⚠️ No data for correlation plot")
                return go.Figure()
            
            df = df.copy()
            if 'temperature_avg' not in df.columns:
                df['temperature_avg'] = df.get('temperature_max', df.get('temperature_min', np.nan))
            
            fig = go.Figure()
            regression_stats = self.data_manager.calculate_regression(df, selected_cities)
            correlations = self.data_manager.calculate_correlations(df, selected_cities)
            
            colors = px.colors.qualitative.Set1
            
            # Store all statistics for prominent display
            all_stats = []
            
            for idx, city in enumerate(selected_cities):
                if city not in df['city'].unique():
                    continue
                    
                city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand'])
                if city_df.empty:
                    continue
                
                color = colors[idx % len(colors)]
                
                # Add scatter points with enhanced hover information
                fig.add_trace(go.Scatter(
                    x=city_df['temperature_avg'],
                    y=city_df['energy_demand'],
                    mode='markers',
                    name=f'{city} Data',
                    marker=dict(color=color, size=8, opacity=0.7, line=dict(width=1, color='white')),
                    text=city_df['date'].dt.strftime('%Y-%m-%d'),
                    customdata=np.column_stack((city_df['temperature_avg'], city_df['energy_demand'])),
                    hovertemplate=(
                        f'<b>{city}</b><br>'
                        '<b>Date:</b> %{text}<br>'
                        '<b>Avg. Temperature:</b> %{x:.1f}°F<br>'
                        '<b>Energy Demand:</b> %{y:,.0f} MWh<br>'
                        '<extra></extra>'
                    )
                ))
                
                # Add regression line with equation
                stats = regression_stats.get(city, {})
                corr_value = correlations.get(city, np.nan)
                
                if not np.isnan(stats.get('slope', np.nan)) and not np.isnan(stats.get('intercept', np.nan)):
                    # Calculate regression line points
                    x_range = [city_df['temperature_avg'].min() - 2, city_df['temperature_avg'].max() + 2]
                    y_pred = [stats['slope'] * x + stats['intercept'] for x in x_range]
                    
                    # Create regression equation string
                    slope = stats['slope']
                    intercept = stats['intercept']
                    if intercept >= 0:
                        equation = f"y = {slope:.2f}x + {intercept:.0f}"
                    else:
                        equation = f"y = {slope:.2f}x - {abs(intercept):.0f}"
                    
                    # Add regression line
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name=f'{city} Regression',
                        line=dict(color=color, dash='dash', width=3),
                        showlegend=True,
                        hovertemplate=(
                            f'<b>{city} Regression Line</b><br>'
                            f'<b>Equation:</b> {equation}<br>'
                            f'<b>R² Value:</b> {stats.get("r_squared", "N/A"):.3f}<br>'
                            f'<b>Correlation (r):</b> {corr_value:.3f}<br>'
                            f'<b>Data Points:</b> {stats.get("data_points", len(city_df))}<br>'
                            '<extra></extra>'
                        )
                    ))
                    
                    # Store stats for summary display
                    all_stats.append({
                        'city': city,
                        'equation': equation,
                        'r_squared': stats.get('r_squared', np.nan),
                        'correlation': corr_value,
                        'slope': slope,
                        'data_points': stats.get('data_points', len(city_df))
                    })
            
            # Create comprehensive statistics annotation
            if show_correlations and all_stats:
                # Sort by R-squared value for better display
                all_stats.sort(key=lambda x: abs(x['r_squared']) if not np.isnan(x['r_squared']) else 0, reverse=True)
                
                stats_text = "<b>📈 REGRESSION ANALYSIS</b><br>"
                stats_text += "━━━━━━━━━━━━━━━━━━━━━━<br>"
                
                for stat in all_stats:
                    if not np.isnan(stat['r_squared']) and not np.isnan(stat['correlation']):
                        # Determine correlation strength
                        corr_abs = abs(stat['correlation'])
                        if corr_abs > 0.7:
                            strength = "Strong"
                            strength_color = "#28a745"  # Green
                        elif corr_abs > 0.4:
                            strength = "Moderate"
                            strength_color = "#ffc107"  # Yellow
                        else:
                            strength = "Weak" 
                            strength_color = "#dc3545"  # Red
                        
                        stats_text += f"<b>{stat['city']}:</b><br>"
                        stats_text += f"  • Equation: {stat['equation']}<br>"
                        stats_text += f"  • R²: {stat['r_squared']:.3f}<br>"
                        stats_text += f"  • Correlation: {stat['correlation']:.3f} ({strength})<br>"
                        stats_text += f"  • Points: {stat['data_points']}<br>"
                        stats_text += "<br>"
                
                # Add the annotation box - moved to far left
                fig.add_annotation(
                    x=0.01, y=0.99,
                    xref='paper', yref='paper',
                    text=stats_text,
                    showarrow=False,
                    bgcolor='rgba(20, 20, 20, 0.95)',
                    bordercolor='#00d4ff',
                    borderwidth=2,
                    borderpad=8,
                    align='left',
                    font=dict(size=10, color='#ffffff'),
                    width=260,
                    valign='top',
                    xanchor='left',
                    yanchor='top'
                )
            
            # Add overall summary statistics if multiple cities
            if len(all_stats) > 1:
                valid_corrs = [s['correlation'] for s in all_stats if not np.isnan(s['correlation'])]
                valid_r2 = [s['r_squared'] for s in all_stats if not np.isnan(s['r_squared'])]
                
                if valid_corrs and valid_r2:
                    summary_text = f"<b>📊 SUMMARY STATISTICS</b><br>"
                    summary_text += f"Cities Analyzed: {len(all_stats)}<br>"
                    summary_text += f"Avg Correlation: {np.mean(valid_corrs):.3f}<br>"
                    summary_text += f"Avg R²: {np.mean(valid_r2):.3f}<br>"
                    summary_text += f"Range (r): {min(valid_corrs):.3f} to {max(valid_corrs):.3f}"
                    
                    fig.add_annotation(
                        x=0.99, y=0.05,
                        xref='paper', yref='paper',
                        text=summary_text,
                        showarrow=False,
                        bgcolor='rgba(20, 20, 20, 0.95)',
                        bordercolor='#00ff88',
                        borderwidth=2,
                        borderpad=6,
                        align='left',
                        font=dict(size=10, color='#ffffff'),
                        xanchor='right',
                        yanchor='bottom'
                    )
            
            # Enhanced layout with dark theme
            fig.update_layout(
                title=dict(
                    text='📊Avg. Temperature vs. Energy Demand - Correlation Analysis',
                    x=0.5,
                    font=dict(size=20, color='#00d4ff', family='Arial Black')
                ),
                xaxis=dict(
                    title=dict(text='🌡️ Average Temperature (°F)', font=dict(size=14, color='#ffffff')),
                    tickfont=dict(size=12, color='#ffffff'),
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    showgrid=True,
                    zeroline=False
                ),
                yaxis=dict(
                    title=dict(text='⚡ Energy Demand (MWh)', font=dict(size=14, color='#ffffff')),
                    tickfont=dict(size=12, color='#ffffff'),
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    showgrid=True,
                    zeroline=False
                ),
                height=600,
                width=1200,
                showlegend=True,
                legend=dict(
                    yanchor="top", 
                    y=0.99, 
                    xanchor="right", 
                    x=0.99,
                    bgcolor='rgba(20, 20, 20, 0.9)',
                    bordercolor='rgba(255, 255, 255, 0.3)',
                    borderwidth=1,
                    font=dict(size=10, color='#ffffff')
                ),
                hovermode='closest',
                paper_bgcolor='rgba(15, 15, 15, 1)',
                plot_bgcolor='rgba(25, 25, 25, 1)',
                margin=dict(l=60, r=60, t=60, b=60),
                font=dict(color='#ffffff')
            )
            
            # Add subtle grid lines with dark theme
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
            
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
                    name=f'{city} Avg. Temperature',
                    line=dict(color=color, width=2),
                    yaxis='y1',
                    hovertemplate=f'<b>{city}</b><br>Date: %{{x}}<br>Avg. Temperature: %{{y:.1f}}°F<extra></extra>'
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
                    text='📈 Avg. Temperature and Energy Demand Timeline',
                    x=0.5,
                    font=dict(size=18, color='#1f77b4')
                ),
                xaxis_title='📅 Date',
                yaxis=dict(title='🌡️ Avg. Temperature (°F)', color='#d62728', side='left'),
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
    
    def create_geographic_map(self, df: pd.DataFrame, selected_cities: List[str], 
                        lookback_days: int, recent_days: int, show_last_day_change: bool) -> go.Figure:
        """Create interactive map visualization."""
        try:
            if df.empty:
                st.warning("⚠️ No data for geographic map")
                return go.Figure()
            
            # Calculate usage levels
            usage_levels = self.data_manager.calculate_usage_levels(
                df, 
                selected_cities=selected_cities,
                lookback_days=lookback_days,
                recent_days=recent_days
            )
            
            if not usage_levels:
                st.warning("⚠️ No usage level data available for mapping")
                return go.Figure()
            
            # Calculate last day's changes if enabled
            last_day_changes = {}
            if show_last_day_change:
                last_day_changes = self.data_manager.calculate_last_day_change(df)
            
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
                    'change_text': change_text,
                    'recent_label': city_data['recent_label'],
                    'baseline_label': city_data['baseline_label'],
                    'recent_period': city_data['recent_period'],
                    'baseline_period': city_data['baseline_period']
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
                        f"⚡ {x['recent_label']}: {x['current_usage']:,.1f} MWh<br>"
                        f"📊 {x['baseline_label']}: {x['baseline_median']:,.1f} MWh<br>"
                        f"🔄 {x['status_description']}<br>"
                        f"{x['change_text']}<br>"
                        f"📅 Recent Period: {x['recent_period']}<br>"
                        f"📅 Baseline Period: {x['baseline_period']}<br>"
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
            
        except Exception as e:
            logger.error(f"Failed to create geographic map: {str(e)}")
            st.error(f"❌ Geographic map failed: {str(e)}")
            return go.Figure()