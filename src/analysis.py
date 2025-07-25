"""Data analysis functions for weather and energy data."""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from config import Config
from loguru import logger
from pathlib import Path

class Analyzer:
    """Base class for data analysis and preprocessing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_fetch_days = self.config.rate_limits.get('max_fetch_days', 90)
        self.buffer_days = self.config.rate_limits.get('buffer_days', 5)
    
    def load_data(self, date_range: Tuple[datetime.date, datetime.date]) -> pd.DataFrame:
        """Load and filter processed data."""
        try:
            data_file = Path(self.config.data_paths['processed']) / "latest_historical.csv"
            if not data_file.exists():
                raise FileNotFoundError("No processed data found at latest_historical.csv")
            
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Apply date range filter
            start_date, end_date = date_range
            mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
            df = df[mask].copy()
            
            # Ensure temperature_avg is available
            if 'temperature_avg' not in df.columns:
                if 'temperature_max' in df.columns and 'temperature_min' in df.columns:
                    df['temperature_avg'] = (df['temperature_max'] + df['temperature_min']) / 2
                elif 'temperature_max' in df.columns:
                    df['temperature_avg'] = df['temperature_max']
                elif 'temperature_min' in df.columns:
                    df['temperature_avg'] = df['temperature_min']
                else:
                    df['temperature_avg'] = np.nan
            
            # Add derived fields
            df['day_of_week'] = df['date'].dt.day_name()
            df['is_weekend'] = df['date'].dt.dayofweek >= 5
            
            logger.info(f"Loaded {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")
            logger.debug(f"Data summary:\n{df.describe()}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_correlations(self, df: pd.DataFrame, selected_cities: List[str]) -> Dict[str, float]:
        """Calculate correlations between temperature and energy demand by city."""
        try:
            if df.empty:
                logger.warning("No data for correlation calculation")
                return {}
            
            df = df.copy()
            if 'temperature_avg' not in df.columns:
                if 'temperature_max' in df.columns and 'temperature_min' in df.columns:
                    df['temperature_avg'] = (df['temperature_max'] + df['temperature_min']) / 2
                elif 'temperature_max' in df.columns:
                    df['temperature_avg'] = df['temperature_max']
                else:
                    df['temperature_avg'] = df.get('temperature_min', np.nan)
            
            correlations = {}
            cities_to_process = selected_cities if selected_cities and 'All Cities' not in selected_cities else df['city'].unique()
            
            for city in cities_to_process:
                city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand'])
                if len(city_df) < 2:
                    logger.warning(f"Insufficient data for correlation in {city}")
                    correlations[city] = np.nan
                    continue
                
                try:
                    corr = city_df['temperature_avg'].corr(city_df['energy_demand'])
                    correlations[city] = round(corr, 3) if not np.isnan(corr) else np.nan
                    logger.debug(f"Correlation for {city}: {corr:.3f}")
                except Exception as corr_error:
                    logger.warning(f"Correlation calculation failed for {city}: {str(corr_error)}")
                    correlations[city] = np.nan
            
            return correlations
            
        except Exception as e:
            logger.error(f"Failed to calculate correlations: {str(e)}")
            return {}
    
    def calculate_regression(self, df: pd.DataFrame, selected_cities: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate linear regression stats for temperature vs. energy demand."""
        try:
            if df.empty:
                logger.warning("No data for regression calculation")
                return {}
            
            df = df.copy()
            if 'temperature_avg' not in df.columns:
                if 'temperature_max' in df.columns and 'temperature_min' in df.columns:
                    df['temperature_avg'] = (df['temperature_max'] + df['temperature_min']) / 2
                elif 'temperature_max' in df.columns:
                    df['temperature_avg'] = df['temperature_max']
                else:
                    df['temperature_avg'] = df.get('temperature_min', np.nan)
            
            regression_stats = {}
            cities_to_process = selected_cities if selected_cities and 'All Cities' not in selected_cities else df['city'].unique()
            
            for city in cities_to_process:
                city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand'])
                if len(city_df) < 2:
                    logger.warning(f"Insufficient data for regression in {city}")
                    regression_stats[city] = {
                        'slope': np.nan, 
                        'intercept': np.nan, 
                        'r_squared': np.nan
                    }
                    continue
                
                try:
                    slope, intercept, r_value, _, _ = linregress(
                        city_df['temperature_avg'],
                        city_df['energy_demand']
                    )
                    regression_stats[city] = {
                        'slope': round(slope, 2),
                        'intercept': round(intercept, 2),
                        'r_squared': round(r_value**2, 3)
                    }
                    logger.debug(f"Regression for {city}: slope={slope:.2f}, intercept={intercept:.2f}, R²={r_value**2:.3f}")
                except Exception as reg_error:
                    logger.warning(f"Regression calculation failed for {city}: {str(reg_error)}")
                    regression_stats[city] = {
                        'slope': np.nan, 
                        'intercept': np.nan, 
                        'r_squared': np.nan
                    }
            
            return regression_stats
            
        except Exception as e:
            logger.error(f"Failed to calculate regression: {str(e)}")
            return {}
    
    def calculate_usage_levels(self, df: pd.DataFrame, selected_cities: List[str] = None, 
                             lookback_days: int = 30, recent_days: int = 7) -> Dict[str, Dict[str, any]]:
        """
        Calculate high/low energy usage levels for geographical mapping.
        
        Args:
            df: DataFrame with energy and city data
            selected_cities: List of cities to analyze (None for all cities)
            lookback_days: Days of historical data to use for baseline calculation
            recent_days: Days of recent data to use for current usage calculation
            
        Returns:
            Dict with city usage levels and coordinates for mapping
        """
        try:
            if df.empty:
                logger.warning("No data for usage level calculation")
                return {}
            
            usage_levels = {}
            cities_to_process = selected_cities if selected_cities and 'All Cities' not in selected_cities else df['city'].unique()
            
            # Get recent date range for current usage calculation
            max_date = df['date'].max()
            recent_cutoff = max_date - pd.Timedelta(days=recent_days)
            baseline_cutoff = max_date - pd.Timedelta(days=lookback_days)
            
            logger.info(f"Calculating usage levels - Recent period: {recent_cutoff.date()} to {max_date.date()}")
            logger.info(f"Baseline period: {baseline_cutoff.date()} to {max_date.date()}")
            
            for city in cities_to_process:
                city_df = df[df['city'] == city].dropna(subset=['energy_demand'])
                if len(city_df) < 10:  # Need sufficient data
                    logger.warning(f"Insufficient data for {city}: {len(city_df)} records")
                    continue
                
                # Get city coordinates from config
                city_config = self.config.get_city_by_name(city)
                if not city_config:
                    logger.warning(f"City configuration not found for {city}")
                    continue
                
                # Calculate baseline statistics using historical data
                baseline_df = city_df[city_df['date'] >= baseline_cutoff]
                if len(baseline_df) < 5:
                    logger.warning(f"Insufficient baseline data for {city}: {len(baseline_df)} records")
                    continue
                
                # Calculate median for threshold (more robust than mean)
                baseline_median = baseline_df['energy_demand'].median()
                baseline_std = baseline_df['energy_demand'].std()
                
                # Calculate current/recent average
                recent_df = city_df[city_df['date'] >= recent_cutoff]
                if len(recent_df) == 0:
                    logger.warning(f"No recent data for {city}")
                    continue
                
                current_usage = recent_df['energy_demand'].mean()
                
                # Determine status using median + standard deviation threshold
                # High usage: above median + 0.5 * std deviation
                # Low usage: below median - 0.5 * std deviation
                high_threshold = baseline_median + (0.5 * baseline_std)
                low_threshold = baseline_median - (0.5 * baseline_std)
                
                if current_usage >= high_threshold:
                    status = 'high'
                    color = '#d62728'  # Red
                    status_description = f"Above average (+{((current_usage - baseline_median) / baseline_median * 100):.1f}%)"
                else:
                    status = 'low'
                    color = '#2ca02c'  # Green  
                    status_description = f"Below average ({((current_usage - baseline_median) / baseline_median * 100):.1f}%)"
                
                usage_levels[city] = {
                    'lat': city_config.lat,
                    'lon': city_config.lon,
                    'current_usage': round(current_usage, 2),
                    'baseline_median': round(baseline_median, 2),
                    'high_threshold': round(high_threshold, 2),
                    'low_threshold': round(low_threshold, 2),
                    'status': status,
                    'color': color,
                    'city_name': city,
                    'state': city_config.state,
                    'status_description': status_description,
                    'recent_data_points': len(recent_df),
                    'baseline_data_points': len(baseline_df),
                    'last_updated': max_date.strftime('%Y-%m-%d')
                }
                
                logger.debug(f"Usage level for {city}: {status} (current: {current_usage:.2f}, median: {baseline_median:.2f})")
            
            logger.info(f"Calculated usage levels for {len(usage_levels)} cities")
            return usage_levels
            
        except Exception as e:
            logger.error(f"Failed to calculate usage levels: {str(e)}")
            return {}
    
    def get_usage_summary(self, usage_levels: Dict[str, Dict[str, any]]) -> Dict[str, any]:
        """
        Generate summary statistics from usage levels calculation.
        
        Args:
            usage_levels: Output from calculate_usage_levels()
            
        Returns:
            Summary statistics dict
        """
        try:
            if not usage_levels:
                return {
                    'total_cities': 0, 
                    'high_count': 0, 
                    'low_count': 0,
                    'high_cities': [],
                    'low_cities': [],
                    'high_usage_avg': 0,
                    'low_usage_avg': 0
                }
            
            high_cities = [city for city, data in usage_levels.items() if data['status'] == 'high']
            low_cities = [city for city, data in usage_levels.items() if data['status'] == 'low']
            
            # Calculate average usage by status
            high_usage = [data['current_usage'] for data in usage_levels.values() if data['status'] == 'high']
            low_usage = [data['current_usage'] for data in usage_levels.values() if data['status'] == 'low']
            
            summary = {
                'total_cities': len(usage_levels),
                'high_count': len(high_cities),
                'low_count': len(low_cities),
                'high_cities': high_cities,
                'low_cities': low_cities,
                'high_usage_avg': round(np.mean(high_usage), 2) if high_usage else 0,
                'low_usage_avg': round(np.mean(low_usage), 2) if low_usage else 0,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Usage summary: {summary['high_count']} high, {summary['low_count']} low usage cities")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate usage summary: {str(e)}")
            return {'error': str(e)}
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate a data quality report."""
        if df.empty:
            return {
                'passed': False,
                'issues': ['No data to check'],
                'summary': {
                    'total_rows': 0, 
                    'missing_values': 0, 
                    'outliers': 0,
                    'data_freshness_days': None
                }
            }
        
        try:
            issues = []
            summary = {
                'total_rows': len(df),
                'missing_values': 0,
                'outliers': 0,
                'data_freshness_days': None
            }
            
            # Check for temperature columns
            temp_columns = [col for col in ['temperature_avg', 'temperature_max', 'temperature_min'] if col in df.columns]
            data_columns = temp_columns + (['energy_demand'] if 'energy_demand' in df.columns else [])
            
            if not data_columns:
                issues.append("No temperature or energy columns found")
                return {
                    'passed': False,
                    'issues': issues,
                    'summary': summary
                }
            
            # Check missing values
            missing_counts = df[data_columns].isnull().sum()
            summary['missing_values'] = int(missing_counts.sum())
            
            for column, count in missing_counts.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    issues.append(f"Missing values in {column}: {count} ({percentage:.1f}%)")
            
            # Check temperature outliers
            for col in temp_columns:
                if col not in df.columns:
                    continue
                    
                temp_data = df[col].dropna()
                if len(temp_data) == 0:
                    continue
                    
                max_temp = self.config.quality_checks.get('temperature', {}).get('max_fahrenheit', 130)
                min_temp = self.config.quality_checks.get('temperature', {}).get('min_fahrenheit', -50)
                
                outlier_mask = (df[col] < min_temp) | (df[col] > max_temp)
                outliers = df[outlier_mask & df[col].notna()]
                summary['outliers'] += len(outliers)
                
                if len(outliers) > 0:
                    issues.append(f"Outliers in {col}: {len(outliers)} values outside [{min_temp}, {max_temp}]°F")
            
            # Check energy outliers
            if 'energy_demand' in df.columns:
                min_energy = self.config.quality_checks.get('energy', {}).get('min_value', 0)
                energy_outliers = df[df['energy_demand'] < min_energy]
                summary['outliers'] += len(energy_outliers)
                if len(energy_outliers) > 0:
                    issues.append(f"Negative energy values: {len(energy_outliers)}")
            
            # Check data freshness
            if 'date' in df.columns and not df['date'].isnull().all():
                latest_date = df['date'].max()
                days_old = (datetime.now() - latest_date).days
                summary['data_freshness_days'] = days_old
                max_age = self.config.quality_checks.get('freshness', {}).get('max_age_hours', 48) / 24
                if days_old > max_age:
                    issues.append(f"Data is stale: {days_old} days old (threshold: {max_age:.1f} days)")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {str(e)}")
            return {
                'passed': False,
                'issues': [f"Quality check error: {str(e)}"],
                'summary': {
                    'total_rows': len(df) if not df.empty else 0,
                    'missing_values': 0,
                    'outliers': 0,
                    'data_freshness_days': None
                }
            }