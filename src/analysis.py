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
        # Use fixed 2-day buffer to match the main pipeline changes
        self.buffer_days = 2
        logger.debug(f"Analyzer initialized with {self.buffer_days}-day buffer")
    
    def get_available_date_range(self) -> Tuple[datetime.date, datetime.date]:
        """Get the available date range considering buffer days."""
        today = datetime.now().date()
        end_date = today - timedelta(days=self.buffer_days)  # 2-day buffer
        start_date = end_date - timedelta(days=self.max_fetch_days - 1)
        
        logger.debug(f"Available date range: {start_date} to {end_date} (today: {today}, buffer: {self.buffer_days} days)")
        return start_date, end_date
    
    def load_data(self, date_range: Optional[Tuple[datetime.date, datetime.date]] = None) -> pd.DataFrame:
        """Load and filter processed data."""
        try:
            data_file = Path(self.config.data_paths['processed']) / "latest_historical.csv"
            if not data_file.exists():
                # Try alternative file names
                alt_files = [
                    "latest_daily.csv",
                    "processed_data.csv"
                ]
                data_file = None
                for alt_file in alt_files:
                    alt_path = Path(self.config.data_paths['processed']) / alt_file
                    if alt_path.exists():
                        data_file = alt_path
                        logger.info(f"Using alternative data file: {alt_file}")
                        break
                
                if data_file is None:
                    raise FileNotFoundError(f"No processed data found. Looked for: latest_historical.csv, {', '.join(alt_files)}")
            
            logger.info(f"Loading data from: {data_file}")
            df = pd.read_csv(data_file)
            
            if df.empty:
                logger.warning("Loaded data file is empty")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} rows from {data_file}")
            
            # Convert date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                logger.error("No 'date' column found in data")
                return pd.DataFrame()
            
            # Apply date range filter if provided, otherwise use available range
            if date_range is None:
                start_date, end_date = self.get_available_date_range()
                logger.info(f"No date range specified, using available range: {start_date} to {end_date}")
            else:
                start_date, end_date = date_range
                logger.info(f"Using specified date range: {start_date} to {end_date}")
            
            # Filter by date range
            initial_count = len(df)
            mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
            df = df[mask].copy()
            filtered_count = len(df)
            
            logger.info(f"Date filtering: {initial_count} -> {filtered_count} records")
            
            if df.empty:
                logger.warning(f"No data available in date range {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Ensure temperature_avg is available
            if 'temperature_avg' not in df.columns:
                logger.info("Creating temperature_avg column from available temperature data")
                if 'temperature_max' in df.columns and 'temperature_min' in df.columns:
                    df['temperature_avg'] = (df['temperature_max'] + df['temperature_min']) / 2
                    logger.debug("Created temperature_avg from temperature_max and temperature_min")
                elif 'temperature_max' in df.columns:
                    df['temperature_avg'] = df['temperature_max']
                    logger.debug("Using temperature_max as temperature_avg")
                elif 'temperature_min' in df.columns:
                    df['temperature_avg'] = df['temperature_min']
                    logger.debug("Using temperature_min as temperature_avg")
                else:
                    df['temperature_avg'] = np.nan
                    logger.warning("No temperature columns found, setting temperature_avg to NaN")
            
            # Add derived fields
            df['day_of_week'] = df['date'].dt.day_name()
            df['is_weekend'] = df['date'].dt.dayofweek >= 5
            
            # Log data summary
            date_range_actual = f"{df['date'].min().date()} to {df['date'].max().date()}"
            cities = df['city'].nunique() if 'city' in df.columns else 0
            
            logger.info(f"Final dataset: {len(df)} records from {date_range_actual} across {cities} cities")
            
            # Log column availability
            required_columns = ['city', 'date', 'temperature_avg', 'energy_demand']
            available_columns = [col for col in required_columns if col in df.columns]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            logger.debug(f"Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return pd.DataFrame()
    
    


    def calculate_correlations(self, df: pd.DataFrame, selected_cities: List[str] = None) -> Dict[str, float]:
        """Calculate correlations between temperature and energy demand by city with improved accuracy."""
        try:
            if df.empty:
                logger.warning("No data for correlation calculation")
                return {}
            
            # Ensure we have required columns
            required_cols = ['city', 'temperature_avg', 'energy_demand']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for correlation: {missing_cols}")
                return {}
            
            df = df.copy()
            
            # Improved temperature_avg handling with better fallback logic
            if df['temperature_avg'].isna().all() or 'temperature_avg' not in df.columns:
                logger.info("Creating temperature_avg from available temperature data")
                if 'temperature_max' in df.columns and 'temperature_min' in df.columns:
                    # Use both min and max for more accurate average
                    df['temperature_avg'] = (df['temperature_max'] + df['temperature_min']) / 2
                    logger.debug("Created temperature_avg from temperature_max and temperature_min")
                elif 'temperature_max' in df.columns:
                    # Use max but apply correction factor (typically 5-8°F higher than average)
                    df['temperature_avg'] = df['temperature_max'] - 5  # Conservative correction
                    logger.debug("Using temperature_max minus 5°F as temperature_avg approximation")
                elif 'temperature_min' in df.columns:
                    # Use min but apply correction factor (typically 10-15°F lower than average)
                    df['temperature_avg'] = df['temperature_min'] + 12  # Conservative correction
                    logger.debug("Using temperature_min plus 12°F as temperature_avg approximation")
                else:
                    df['temperature_avg'] = np.nan
                    logger.warning("No temperature columns found, setting temperature_avg to NaN")
            
            correlations = {}
            
            # Determine cities to process
            if not selected_cities or 'All Cities' in selected_cities:
                cities_to_process = df['city'].unique()
                logger.info(f"Processing correlations for all cities: {list(cities_to_process)}")
            else:
                cities_to_process = selected_cities
                logger.info(f"Processing correlations for selected cities: {cities_to_process}")
            
            for city in cities_to_process:
                city_df = df[df['city'] == city].copy()
                
                # Remove outliers that could skew correlation (using IQR method)
                for col in ['temperature_avg', 'energy_demand']:
                    if col in city_df.columns:
                        Q1 = city_df[col].quantile(0.25)
                        Q3 = city_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Log outliers before removal
                        outliers = city_df[(city_df[col] < lower_bound) | (city_df[col] > upper_bound)]
                        if len(outliers) > 0:
                            logger.debug(f"Removing {len(outliers)} outliers in {col} for {city}")
                        
                        city_df = city_df[(city_df[col] >= lower_bound) & (city_df[col] <= upper_bound)]
                
                # Final data cleaning
                city_df = city_df.dropna(subset=['temperature_avg', 'energy_demand'])
                
                # Require minimum data points for reliable correlation
                min_points = max(10, len(df[df['city'] == city]) * 0.7)  # At least 10 points or 70% of original data
                if len(city_df) < min_points:
                    logger.warning(f"Insufficient data for reliable correlation in {city}: {len(city_df)} records (need {min_points:.0f})")
                    correlations[city] = np.nan
                    continue
                
                try:
                    # Calculate Pearson correlation with significance test
                    from scipy.stats import pearsonr
                    corr_coef, p_value = pearsonr(city_df['temperature_avg'], city_df['energy_demand'])
                    
                    # Only use correlation if it's statistically significant (p < 0.05)
                    if p_value < 0.05:
                        correlations[city] = round(corr_coef, 3)
                        logger.debug(f"Correlation for {city}: {corr_coef:.3f} (p={p_value:.4f}, n={len(city_df)}) - Significant")
                    else:
                        correlations[city] = np.nan
                        logger.debug(f"Correlation for {city}: {corr_coef:.3f} (p={p_value:.4f}, n={len(city_df)}) - Not significant")
                        
                except Exception as corr_error:
                    logger.warning(f"Correlation calculation failed for {city}: {str(corr_error)}")
                    correlations[city] = np.nan
            
            # Log summary statistics
            valid_corrs = [v for v in correlations.values() if not np.isnan(v)]
            if valid_corrs:
                logger.info(f"Calculated {len(valid_corrs)} significant correlations. Range: {min(valid_corrs):.3f} to {max(valid_corrs):.3f}")
            
            return correlations
            
        except Exception as e:
            logger.error(f"Failed to calculate correlations: {str(e)}")
            return {}
        
    def calculate_regression(self, df: pd.DataFrame, selected_cities: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Calculate linear regression stats for temperature vs. energy demand."""
        try:
            if df.empty:
                logger.warning("No data for regression calculation")
                return {}
            
            # Ensure we have required columns
            required_cols = ['city', 'temperature_avg', 'energy_demand']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for regression: {missing_cols}")
                return {}
            
            df = df.copy()
            
            # Handle temperature_avg creation if needed
            if df['temperature_avg'].isna().all():
                logger.info("All temperature_avg values are NaN, attempting to recreate from other columns")
                if 'temperature_max' in df.columns and 'temperature_min' in df.columns:
                    df['temperature_avg'] = (df['temperature_max'] + df['temperature_min']) / 2
                elif 'temperature_max' in df.columns:
                    df['temperature_avg'] = df['temperature_max']
                elif 'temperature_min' in df.columns:
                    df['temperature_avg'] = df['temperature_min']
            
            regression_stats = {}
            
            # Determine cities to process
            if not selected_cities or 'All Cities' in selected_cities:
                cities_to_process = df['city'].unique()
                logger.info(f"Processing regression for all cities: {list(cities_to_process)}")
            else:
                cities_to_process = selected_cities
                logger.info(f"Processing regression for selected cities: {cities_to_process}")
            
            for city in cities_to_process:
                city_df = df[df['city'] == city].dropna(subset=['temperature_avg', 'energy_demand'])
                
                if len(city_df) < 2:
                    logger.warning(f"Insufficient data for regression in {city}: {len(city_df)} records")
                    regression_stats[city] = {
                        'slope': np.nan, 
                        'intercept': np.nan, 
                        'r_squared': np.nan
                    }
                    continue
                
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        city_df['temperature_avg'],
                        city_df['energy_demand']
                    )
                    regression_stats[city] = {
                        'slope': round(slope, 2),
                        'intercept': round(intercept, 2),
                        'r_squared': round(r_value**2, 3),
                        'p_value': round(p_value, 4),
                        'data_points': len(city_df)
                    }
                    logger.debug(f"Regression for {city}: slope={slope:.2f}, intercept={intercept:.2f}, R²={r_value**2:.3f} (n={len(city_df)})")
                except Exception as reg_error:
                    logger.warning(f"Regression calculation failed for {city}: {str(reg_error)}")
                    regression_stats[city] = {
                        'slope': np.nan, 
                        'intercept': np.nan, 
                        'r_squared': np.nan,
                        'p_value': np.nan,
                        'data_points': len(city_df)
                    }
            
            logger.info(f"Calculated regression stats for {len(regression_stats)} cities")
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
            
            # Ensure we have required columns
            required_cols = ['city', 'date', 'energy_demand']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for usage levels: {missing_cols}")
                return {}
            
            usage_levels = {}
            
            # Determine cities to process
            if not selected_cities or 'All Cities' in selected_cities:
                cities_to_process = df['city'].unique()
                logger.info(f"Processing usage levels for all cities: {list(cities_to_process)}")
            else:
                cities_to_process = selected_cities
                logger.info(f"Processing usage levels for selected cities: {cities_to_process}")
            
            # Get recent date range for current usage calculation
            max_date = df['date'].max()
            recent_cutoff = max_date - pd.Timedelta(days=recent_days)
            baseline_cutoff = max_date - pd.Timedelta(days=lookback_days)
            
            logger.info(f"Usage level calculation periods:")
            logger.info(f"  Recent period: {recent_cutoff.date()} to {max_date.date()} ({recent_days} days)")
            logger.info(f"  Baseline period: {baseline_cutoff.date()} to {max_date.date()} ({lookback_days} days)")
            
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
                    'low_usage_avg': 0,
                    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
            
            # Check data freshness (considering buffer days)
            if 'date' in df.columns and not df['date'].isnull().all():
                latest_date = df['date'].max()
                expected_latest = datetime.now() - timedelta(days=self.buffer_days)
                days_behind = (expected_latest - latest_date).days
                summary['data_freshness_days'] = days_behind
                
                max_age = self.config.quality_checks.get('freshness', {}).get('max_age_hours', 48) / 24
                if days_behind > max_age:
                    issues.append(f"Data is stale: {days_behind} days behind expected date (threshold: {max_age:.1f} days)")
                else:
                    logger.debug(f"Data freshness OK: {days_behind} days behind expected (within {max_age:.1f} day threshold)")
            
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