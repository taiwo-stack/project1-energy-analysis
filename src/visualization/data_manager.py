"""Data management and processing utilities."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from streamlit.runtime.caching import cache_data
import streamlit as st
from loguru import logger
from analysis import Analyzer


class DataManager(Analyzer):
    """Handles data loading, caching, and processing."""
    
    def __init__(self, config):
        super().__init__(config)
    
    @cache_data(ttl=3600, show_spinner="🔄 Loading data...")
    def load_cached_data(_self, date_range: Tuple[datetime.date, datetime.date]) -> pd.DataFrame:
        """Load and cache processed data using Analyzer."""
        return _self.load_data(date_range)
    
    def filter_data(self, df: pd.DataFrame, selected_cities: List[str]) -> pd.DataFrame:
        """Filter data based on user selections."""
        if selected_cities:
            df = df[df['city'].isin(selected_cities)]
        return df
    
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
    
    def calculate_last_day_change(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate last recorded day's energy usage % change from the previous day for all cities."""
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided for last day change calculation")
                return {}
            
            changes = {}
            
            for city in df['city'].unique():
                city_df = df[df['city'] == city].sort_values('date')
                
                if len(city_df) < 2:
                    logger.debug(f"Insufficient data for {city}: only {len(city_df)} records")
                    continue
                
                # Get the last two available data points for this city
                last_two = city_df.tail(2)
                
                if len(last_two) == 2:
                    latest_usage = last_two.iloc[-1]['energy_demand']
                    previous_usage = last_two.iloc[-2]['energy_demand']
                    latest_date = last_two.iloc[-1]['date']
                    previous_date = last_two.iloc[-2]['date']
                    
                    # Additional validation for realistic energy values
                    if pd.isna(latest_usage) or pd.isna(previous_usage):
                        logger.warning(f"Missing energy data for {city}")
                        continue
                    
                    if latest_usage < 0 or previous_usage < 0:
                        logger.warning(f"Negative energy values detected for {city}")
                        continue
                    
                    if previous_usage == 0:
                        # Handle division by zero case
                        if latest_usage == 0:
                            pct_change = 0.0
                            logger.debug(f"Both values are zero for {city}")
                        else:
                            # When previous is 0 but current isn't, it's infinite % increase
                            # We'll represent this as a very large number or special case
                            pct_change = float('inf')
                            logger.warning(f"Previous usage was zero for {city}, infinite change")
                            continue  # Skip infinite changes
                    else:
                        # Standard percentage change calculation
                        pct_change = ((latest_usage - previous_usage) / previous_usage) * 100
                    
                    days_between = (latest_date - previous_date).days
                    
                    # Validate the time gap is reasonable (not too large)
                    if days_between > 7:
                        logger.warning(f"Large gap between data points for {city}: {days_between} days")
                    
                    changes[city] = {
                        'latest_usage': round(float(latest_usage), 2),
                        'previous_usage': round(float(previous_usage), 2),
                        'absolute_change': round(float(latest_usage - previous_usage), 2),
                        'pct_change': round(float(pct_change), 1),
                        'latest_date': latest_date.strftime('%Y-%m-%d'),
                        'previous_date': previous_date.strftime('%Y-%m-%d'),
                        'days_between': days_between
                    }
                    
                    logger.debug(f"Calculated change for {city}: {pct_change:.1f}% "
                            f"({previous_usage:.2f} → {latest_usage:.2f})")
            
            logger.info(f"Successfully calculated changes for {len(changes)} cities")
            return changes
            
        except Exception as e:
            logger.error(f"Failed to calculate last day's change: {str(e)}")
            return {}

            
    def prepare_historical_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
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


    def calculate_weather_change(self, df: pd.DataFrame) -> dict:
        """Calculate weather changes similar to energy changes."""
        changes = {}
        
        for city in df['city'].unique():
            city_data = df[df['city'] == city].sort_values('date')
            
            if len(city_data) >= 2:
                latest = city_data.iloc[-1]
                previous = city_data.iloc[-2]
                
                temp_change = latest['temperature_avg'] - previous['temperature_avg']
                days_between = (latest['date'] - previous['date']).days
                
                changes[city] = {
                    'latest_temp': latest['temperature_avg'],
                    'previous_temp': previous['temperature_avg'],
                    'temp_change': temp_change,
                    'latest_date': latest['date'].strftime('%Y-%m-%d'),
                    'previous_date': previous['date'].strftime('%Y-%m-%d'),
                    'days_between': days_between
                }
        
        return changes    
        

class DataQualityChecker:
    """Handles data quality assessment and reporting."""
    
    @staticmethod
    def show_data_quality(df: pd.DataFrame, show_quality: bool, data_manager: DataManager):
        """Display enhanced data quality report including data freshness."""
        if not show_quality or df.empty:
            return
        
        with st.expander("🔍 Data Quality Report"):
            quality_report = data_manager.generate_data_quality_report(df)
            summary = quality_report['summary']
            
            # Check data freshness
            freshness_info = data_manager.check_data_freshness(df)
            
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
                st.markdown("#### 💡 Insight")
                if freshness_info['days_old'] > 7:
                    st.write("• 📞 **Action:** Check data pipeline and API connections")
                elif freshness_info['days_old'] > 3:
                    st.write("• 📈 **Impact:** Recent trends may not be fully captured")
                else:
                    st.write("• 🔄 **Stale reason:** Data is 2 days delayed to wait for data availability before fetching")