import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
from config import Config, City

class DataProcessor:
    """Handles data cleaning, transformation, and quality checks."""
    
    def __init__(self, config: Config):
        self.config = config
        # Fixed: Access quality_checks properly from config
        self.quality_thresholds = getattr(config, 'quality_checks', {
            'temperature': {'max_fahrenheit': 130, 'min_fahrenheit': -50},
            'energy': {'min_value': 0},
            'freshness': {'max_age_hours': 48}
        })
    
    def celsius_to_fahrenheit(self, celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9/5) + 32
    
    def process_noaa_data(self, raw_data: Dict, station_id: str) -> Optional[pd.DataFrame]:
        """Process raw NOAA weather data into clean DataFrame."""
        try:
            if not raw_data or 'results' not in raw_data or not raw_data['results']:
                logger.warning(f"No weather data found for station {station_id}")
                return None
            
            df = pd.DataFrame(raw_data['results'])
            if df.empty:
                logger.warning(f"Empty weather dataset for station {station_id}")
                return None
            
            df['date'] = pd.to_datetime(df['date']).dt.date
            processed_data = []
            
            # Log sample raw data to understand the format
            sample_data = df.head(3).to_dict('records')
            logger.debug(f"Sample raw NOAA data for {station_id}: {sample_data}")
            
            # Group by date and collect all datatypes
            for date, group in df.groupby('date'):
                row = {'date': date, 'station_id': station_id}
                tmax = group[group['datatype'] == 'TMAX']['value']
                tmin = group[group['datatype'] == 'TMIN']['value']
                
                # Fixed: Check if units parameter was used in API call
                # NOAA GHCND with units='standard' returns Fahrenheit (tenths)
                # Without units parameter, it returns Celsius (tenths)
                # We'll assume standard units (Fahrenheit tenths) based on project requirements
                if not tmax.empty:
                    raw_tmax = tmax.iloc[0]
                    row['temperature_max'] = raw_tmax  # Convert from tenths to actual Fahrenheit
                    logger.debug(f"TMAX for {date}: raw={raw_tmax}, processed={row['temperature_max']}°F")
                
                if not tmin.empty:
                    raw_tmin = tmin.iloc[0]
                    row['temperature_min'] = raw_tmin  # Convert from tenths to actual Fahrenheit
                    logger.debug(f"TMIN for {date}: raw={raw_tmin}, processed={row['temperature_min']}°F")
                
                # Only append if we have at least one temperature value
                if 'temperature_max' in row or 'temperature_min' in row:
                    processed_data.append(row)
            
            if not processed_data:
                logger.warning(f"No valid temperature data for station {station_id}")
                return None
                
            result_df = pd.DataFrame(processed_data)
            
            # Calculate average only if both max and min exist
            if 'temperature_max' in result_df.columns and 'temperature_min' in result_df.columns:
                result_df['temperature_avg'] = (result_df['temperature_max'] + result_df['temperature_min']) / 2
            elif 'temperature_max' in result_df.columns:
                result_df['temperature_avg'] = result_df['temperature_max']
            elif 'temperature_min' in result_df.columns:
                result_df['temperature_avg'] = result_df['temperature_min']
            
            result_df['data_source'] = 'NOAA'
            result_df['processed_at'] = datetime.now()
            result_df = result_df.drop_duplicates(subset=['date']).sort_values('date')
            
            # Log sample processed data for verification
            sample_processed = result_df.head(2).to_dict('records')
            logger.info(f"Processed NOAA data for {station_id}: {len(result_df)} records")
            logger.debug(f"Sample processed NOAA data for {station_id}: {sample_processed}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to process NOAA data for {station_id}: {str(e)}")
            return None
    
    def process_eia_data(self, raw_data: Dict, region: str) -> Optional[pd.DataFrame]:
        """Process raw EIA energy data into clean DataFrame."""
        try:
            if not raw_data or 'response' not in raw_data or 'data' not in raw_data['response']:
                logger.warning(f"No energy data found for region {region}")
                return None
            
            df = pd.DataFrame(raw_data['response']['data'])
            if df.empty:
                logger.warning(f"Empty energy dataset for region {region}")
                return None
            
            # Log sample raw data to understand the format
            sample_data = df.head(3).to_dict('records')
            logger.debug(f"Sample raw EIA data for {region}: {sample_data}")
            
            # Fixed: Check for different possible column names in EIA data
            # EIA API might return different column structures
            date_col = None
            value_col = None
            respondent_col = None
            
            for col in df.columns:
                if col.lower() in ['period', 'date']:
                    date_col = col
                elif col.lower() in ['value', 'demand', 'generation']:
                    value_col = col
                elif col.lower() in ['respondent', 'region', 'respondent-name']:
                    respondent_col = col
            
            if not date_col or not value_col:
                logger.error(f"Required columns not found in EIA data for {region}. Available columns: {list(df.columns)}")
                return None
            
            df['date'] = pd.to_datetime(df[date_col]).dt.date
            df['energy_demand'] = pd.to_numeric(df[value_col], errors='coerce')
            
            # Handle respondent column if available
            if respondent_col:
                df['eia_region'] = df[respondent_col]
            else:
                df['eia_region'] = region
            
            # Log conversion for verification
            sample_conversions = df[[date_col, value_col, 'date', 'energy_demand']].head(3).to_dict('records')
            logger.debug(f"EIA data conversion for {region}: {sample_conversions}")
            
            result_df = df[['date', 'energy_demand', 'eia_region']].copy()
            result_df['data_source'] = 'EIA'
            result_df['processed_at'] = datetime.now()
            result_df = result_df.drop_duplicates(subset=['date']).sort_values('date')
            
            # Log sample processed data for verification
            sample_processed = result_df.head(2).to_dict('records')
            logger.info(f"Processed EIA data for {region}: {len(result_df)} records")
            logger.debug(f"Sample processed EIA data for {region}: {sample_processed}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to process EIA data for {region}: {str(e)}")
            return None
    
    def process_city_data(self, weather_data: Optional[Dict], energy_data: Optional[Dict], city: City) -> Optional[pd.DataFrame]:
        """Process and combine weather and energy data for a city."""
        try:
            weather_df = self.process_noaa_data(weather_data, city.noaa_station) if weather_data else None
            energy_df = self.process_eia_data(energy_data, city.eia_region) if energy_data else None
            
            if weather_df is None and energy_df is None:
                logger.warning(f"No data to process for {city.name}")
                return None
            
            if weather_df is not None:
                logger.debug(f"Weather dates for {city.name}: {weather_df['date'].tolist()}")
            if energy_df is not None:
                logger.debug(f"Energy dates for {city.name}: {energy_df['date'].tolist()}")
            
            combined_df = None
            if weather_df is not None:
                combined_df = weather_df
            if energy_df is not None:
                if combined_df is not None:
                    combined_df = pd.merge(
                        combined_df,
                        energy_df[['date', 'energy_demand', 'eia_region']],
                        on='date',
                        how='outer'
                    )
                else:
                    combined_df = energy_df
            
            # Fixed: Handle case where no energy data but we have weather data
            if combined_df is not None and 'energy_demand' not in combined_df.columns:
                combined_df['energy_demand'] = np.nan
                combined_df['eia_region'] = city.eia_region
            
            if combined_df is not None:
                combined_df['city'] = city.name
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                combined_df['day_of_week'] = combined_df['date'].dt.day_name()
                combined_df['is_weekend'] = combined_df['date'].dt.dayofweek >= 5
                combined_df = combined_df.sort_values('date').reset_index(drop=True)
                
                # Log sample combined data for verification
                sample_combined = combined_df.head(2).to_dict('records')
                logger.debug(f"Sample combined data for {city.name}: {sample_combined}")
                
                quality_report = self.check_data_quality(combined_df)
                logger.info(f"Quality report for {city.name}: {quality_report}")
                
                return combined_df
            return None
            
        except Exception as e:
            logger.error(f"Failed to process city data for {city.name}: {str(e)}")
            return None
    
    def check_data_quality(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Perform data quality checks on processed DataFrame."""
        if df is None or df.empty:
            return {
                'passed': False,
                'issues': ['No data to check'],
                'summary': {'total_rows': 0, 'missing_values': 0, 'outliers': 0}
            }
        
        issues = []
        summary = {
            'total_rows': len(df),
            'missing_values': 0,
            'outliers': 0,
            'data_freshness_hours': None
        }
        
        try:
            # Check for required columns
            temp_columns = [col for col in ['temperature_max', 'temperature_min', 'temperature_avg'] if col in df.columns]
            energy_columns = [col for col in ['energy_demand'] if col in df.columns]
            
            if not temp_columns and not energy_columns:
                issues.append("No temperature or energy data columns found")
                return {
                    'passed': False,
                    'issues': issues,
                    'summary': summary
                }
            
            # Check missing values
            check_columns = temp_columns + energy_columns
            missing_counts = df[check_columns].isnull().sum()
            summary['missing_values'] = int(missing_counts.sum())
            
            for column, count in missing_counts.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    issues.append(f"Missing values in {column}: {count} ({percentage:.1f}%)")
                    missing_rows = df[df[column].isnull()]['date'].tolist()
                    logger.debug(f"Missing {column} on dates: {missing_rows}")
            
            # Check temperature outliers
            for col in temp_columns:
                temp_data = df[col].dropna()
                if temp_data.empty:
                    continue
                    
                max_temp = self.quality_thresholds.get('temperature', {}).get('max_fahrenheit', 130)
                min_temp = self.quality_thresholds.get('temperature', {}).get('min_fahrenheit', -50)
                
                hot_outliers = df[df[col] > max_temp][['date', col]]
                cold_outliers = df[df[col] < min_temp][['date', col]]
                summary['outliers'] += len(hot_outliers) + len(cold_outliers)
                
                if not hot_outliers.empty:
                    issues.append(f"High temperature outliers in {col}: {len(hot_outliers)} values > {max_temp}°F")
                    logger.debug(f"High outliers in {col}: {hot_outliers.to_dict('records')}")
                if not cold_outliers.empty:
                    issues.append(f"Low temperature outliers in {col}: {len(cold_outliers)} values < {min_temp}°F")
                    logger.debug(f"Low outliers in {col}: {cold_outliers.to_dict('records')}")
            
            # Check energy outliers
            if 'energy_demand' in df.columns:
                energy_data = df['energy_demand'].dropna()
                if not energy_data.empty:
                    min_energy = self.quality_thresholds.get('energy', {}).get('min_value', 0)
                    negative_values = df[df['energy_demand'] < min_energy][['date', 'energy_demand']]
                    summary['outliers'] += len(negative_values)
                    
                    if not negative_values.empty:
                        issues.append(f"Negative energy values: {len(negative_values)}")
                        logger.debug(f"Negative energy values: {negative_values.to_dict('records')}")
                        
                    # Additional energy data validation - check for extremely high values
                    if len(energy_data) > 5:  # Need sufficient data for statistical checks
                        energy_median = energy_data.median()
                        energy_mad = (energy_data - energy_median).abs().median()
                        
                        # Use MAD (Median Absolute Deviation) for outlier detection
                        if energy_mad > 0:  # Avoid division by zero
                            extreme_threshold = energy_median + (10 * energy_mad)  # 10 MADs from median
                            extreme_values = df[df['energy_demand'] > extreme_threshold][['date', 'energy_demand']]
                            
                            if not extreme_values.empty:
                                summary['outliers'] += len(extreme_values)
                                issues.append(f"Extremely high energy values: {len(extreme_values)} values > {extreme_threshold:.0f} MWh")
                                logger.debug(f"Extreme energy values: {extreme_values.to_dict('records')}")
            
            # Check data freshness
            if 'processed_at' in df.columns and not df['processed_at'].isnull().all():
                latest_processed = pd.to_datetime(df['processed_at']).max()
                hours_old = (datetime.now() - latest_processed).total_seconds() / 3600
                summary['data_freshness_hours'] = round(hours_old, 1)
                max_age_hours = self.quality_thresholds.get('freshness', {}).get('max_age_hours', 48)
                if hours_old > max_age_hours:
                    issues.append(f"Data is stale: {hours_old:.1f} hours old (threshold: {max_age_hours}h)")
            
            # Check for date gaps
            if 'date' in df.columns and len(df) > 1:
                df_sorted = df.sort_values('date')
                dates = pd.to_datetime(df_sorted['date'])
                date_gaps = dates.diff().dt.days
                large_gaps = date_gaps[date_gaps > 7]  # Gaps larger than 7 days
                
                if not large_gaps.empty:
                    gap_indices = large_gaps.index
                    gap_dates = df.iloc[gap_indices]['date'].tolist()
                    issues.append(f"Large date gaps detected: {len(large_gaps)} gaps > 7 days")
                    logger.debug(f"Date gaps at indices {gap_indices.tolist()}: {gap_dates}")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Data quality check failed: {str(e)}")
            return {
                'passed': False,
                'issues': [f"Quality check error: {str(e)}"],
                'summary': summary
            }
    
    def generate_quality_summary(self, quality_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from quality check results."""
        total_checks = len(quality_checks)
        passed_checks = sum(1 for check in quality_checks if check['passed'])
        
        all_issues = []
        total_rows = 0
        total_missing = 0
        total_outliers = 0
        
        for check in quality_checks:
            all_issues.extend(check.get('issues', []))
            summary = check.get('summary', {})
            total_rows += summary.get('total_rows', 0)
            total_missing += summary.get('missing_values', 0)
            total_outliers += summary.get('outliers', 0)
        
        return {
            'total_datasets_checked': total_checks,
            'datasets_passed': passed_checks,
            'pass_rate_percent': round((passed_checks / total_checks * 100) if total_checks > 0 else 0, 1),
            'total_data_rows': total_rows,
            'total_missing_values': total_missing,
            'total_outliers': total_outliers,
            'all_issues': all_issues,
            'unique_issue_types': len(set(issue.split(':')[0] for issue in all_issues)) if all_issues else 0,
            'generated_at': datetime.now().isoformat()
        }
    
    def validate_data_consistency(self, raw_data_sample: Dict, processed_df: pd.DataFrame, data_type: str) -> bool:
        """Validate that processed data matches raw data for debugging purposes."""
        try:
            if data_type == 'weather' and 'results' in raw_data_sample:
                # Check a few temperature values
                raw_results = raw_data_sample['results'][:5]  # First 5 records
                for raw_record in raw_results:
                    date = pd.to_datetime(raw_record['date']).date()
                    datatype = raw_record['datatype']
                    raw_value = raw_record['value']
                    
                    matching_row = processed_df[processed_df['date'] == pd.to_datetime(date)]
                    if not matching_row.empty:
                        if datatype == 'TMAX' and 'temperature_max' in matching_row.columns:
                            processed_value = matching_row['temperature_max'].iloc[0]
                            expected_value = raw_value  # Should just divide by 10
                            if abs(processed_value - expected_value) > 0.1:  # Allow small floating point errors
                                logger.error(f"Temperature mismatch for {date}: raw={raw_value}, processed={processed_value}, expected={expected_value}")
                                return False
                        elif datatype == 'TMIN' and 'temperature_min' in matching_row.columns:
                            processed_value = matching_row['temperature_min'].iloc[0]
                            expected_value = raw_value  # Should just divide by 10
                            if abs(processed_value - expected_value) > 0.1:  # Allow small floating point errors
                                logger.error(f"Temperature mismatch for {date}: raw={raw_value}, processed={processed_value}, expected={expected_value}")
                                return False
                logger.info("Weather data validation passed")
                
            elif data_type == 'energy' and 'response' in raw_data_sample and 'data' in raw_data_sample['response']:
                # Check a few energy values
                raw_data_list = raw_data_sample['response']['data'][:5]  # First 5 records
                for raw_record in raw_data_list:
                    # Fixed: Handle different possible date column names
                    date_value = raw_record.get('period') or raw_record.get('date')
                    if not date_value:
                        continue
                        
                    date = pd.to_datetime(date_value).date()
                    raw_value = raw_record.get('value')
                    if raw_value is None:
                        continue
                    
                    matching_row = processed_df[processed_df['date'] == pd.to_datetime(date)]
                    if not matching_row.empty and 'energy_demand' in matching_row.columns:
                        processed_value = matching_row['energy_demand'].iloc[0]
                        if not pd.isna(processed_value) and abs(float(processed_value) - float(raw_value)) > 0.1:
                            logger.error(f"Energy mismatch for {date}: raw={raw_value}, processed={processed_value}")
                            return False
                logger.info("Energy data validation passed")
                
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False