"""Data handling and processing utilities for the dashboard."""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from config import Config
from loguru import logger


class DataHandler:
    """Handles data processing, filtering, and quality checks."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def filter_data(self, df: pd.DataFrame, selected_cities: List[str]) -> pd.DataFrame:
        """Filter data based on user selections."""
        if selected_cities:
            df = df[df['city'].isin(selected_cities)]
        return df
    
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
    
    def check_data_freshness(self, df: pd.DataFrame) -> Dict[str, Any]:
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
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        try:
            if df.empty:
                return {
                    'summary': {
                        'total_rows': 0,
                        'missing_values': 0,
                        'outliers': 0
                    },
                    'issues': ['No data available for quality assessment']
                }
            
            # Basic statistics
            total_rows = len(df)
            missing_values = df.isnull().sum().sum()
            
            # Detect outliers using IQR method for numeric columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            outliers_count = 0
            
            for col in numeric_columns:
                if col in ['energy_demand', 'temperature_avg', 'temperature_max', 'temperature_min']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_count += len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            
            # Identify issues
            issues = []
            
            if missing_values > 0:
                missing_pct = (missing_values / (total_rows * len(df.columns))) * 100
                issues.append(f"Missing values detected: {missing_values} ({missing_pct:.1f}%)")
            
            if outliers_count > 0:
                outlier_pct = (outliers_count / total_rows) * 100
                issues.append(f"Potential outliers detected: {outliers_count} ({outlier_pct:.1f}%)")
            
            # Check for duplicate records
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                issues.append(f"Duplicate records found: {duplicates}")
            
            # Check date consistency
            if 'date' in df.columns:
                date_gaps = self._check_date_gaps(df)
                if date_gaps:
                    issues.append(f"Date gaps detected: {len(date_gaps)} gaps found")
            
            return {
                'summary': {
                    'total_rows': total_rows,
                    'missing_values': missing_values,
                    'outliers': outliers_count,
                    'duplicates': duplicates
                },
                'issues': issues,
                'date_gaps': date_gaps if 'date' in df.columns else []
            }
            
        except Exception as e:
            logger.error(f"Failed to generate data quality report: {str(e)}")
            return {
                'summary': {
                    'total_rows': 0,
                    'missing_values': 0,
                    'outliers': 0
                },
                'issues': [f'Error generating quality report: {str(e)}']
            }
    
    def _check_date_gaps(self, df: pd.DataFrame) -> List[str]:
        """Check for gaps in the date sequence."""
        try:
            if 'date' not in df.columns or df.empty:
                return []
            
            # Convert date column to datetime if it isn't already
            df_dates = pd.to_datetime(df['date']).dt.date
            
            # Get unique dates and sort them
            unique_dates = sorted(df_dates.unique())
            
            if len(unique_dates) < 2:
                return []
            
            gaps = []
            for i in range(1, len(unique_dates)):
                current_date = unique_dates[i]
                previous_date = unique_dates[i-1]
                
                # Check if there's more than 1 day gap
                gap_days = (current_date - previous_date).days
                if gap_days > 1:
                    gaps.append(f"Gap of {gap_days} days between {previous_date} and {current_date}")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to check date gaps: {str(e)}")
            return []
    
    def validate_data_completeness(self, df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        """Validate that required columns exist and have data."""
        try:
            missing_columns = [col for col in required_columns if col not in df.columns]
            empty_columns = [col for col in required_columns if col in df.columns and df[col].isnull().all()]
            
            is_valid = len(missing_columns) == 0 and len(empty_columns) == 0
            
            validation_result = {
                'is_valid': is_valid,
                'missing_columns': missing_columns,
                'empty_columns': empty_columns,
                'total_rows': len(df),
                'valid_rows': len(df.dropna(subset=[col for col in required_columns if col in df.columns]))
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {
                'is_valid': False,
                'missing_columns': required_columns,
                'empty_columns': [],
                'total_rows': 0,
                'valid_rows': 0,
                'error': str(e)
            }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data for analysis."""
        try:
            if df.empty:
                return df
            
            cleaned_df = df.copy()
            
            # Remove duplicates
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_df)
            
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate rows")
            
            # Handle missing values
            # Forward fill temperature values within each city
            if 'temperature_avg' in cleaned_df.columns:
                cleaned_df['temperature_avg'] = cleaned_df.groupby('city')['temperature_avg'].fillna(method='ffill')
            
            # Remove rows with missing critical data
            critical_columns = ['city', 'date', 'energy_demand']
            cleaned_df = cleaned_df.dropna(subset=critical_columns)
            
            # Ensure date column is proper datetime
            if 'date' in cleaned_df.columns:
                cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
            
            # Sort by city and date
            cleaned_df = cleaned_df.sort_values(['city', 'date'])
            
            logger.info(f"Data cleaning completed. Rows: {initial_rows} -> {len(cleaned_df)}")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return df  # Return original data if cleaning fails
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary statistics."""
        try:
            if df.empty:
                return {'error': 'No data available'}
            
            summary = {
                'basic_info': {
                    'total_records': len(df),
                    'cities_count': df['city'].nunique() if 'city' in df.columns else 0,
                    'date_range': {
                        'start': df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else None,
                        'end': df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else None,
                        'days': (df['date'].max() - df['date'].min()).days if 'date' in df.columns else 0
                    }
                },
                'energy_stats': {},
                'temperature_stats': {},
                'cities': list(df['city'].unique()) if 'city' in df.columns else []
            }
            
            # Energy statistics
            if 'energy_demand' in df.columns:
                summary['energy_stats'] = {
                    'mean': df['energy_demand'].mean(),
                    'median': df['energy_demand'].median(),
                    'std': df['energy_demand'].std(),
                    'min': df['energy_demand'].min(),
                    'max': df['energy_demand'].max(),
                    'total': df['energy_demand'].sum()
                }
            
            # Temperature statistics
            temp_col = 'temperature_avg' if 'temperature_avg' in df.columns else 'temperature_max'
            if temp_col in df.columns:
                summary['temperature_stats'] = {
                    'mean': df[temp_col].mean(),
                    'median': df[temp_col].median(),
                    'std': df[temp_col].std(),
                    'min': df[temp_col].min(),
                    'max': df[temp_col].max()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate data summary: {str(e)}")
            return {'error': str(e)}