"""Main pipeline orchestration for fetching, processing, and analyzing data."""

from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import os
import json
from data_fetcher import DataFetcher
from data_processor import DataProcessor
from config import Config
import time

# Constants for consistency
MAX_FETCH_DAYS = 90
BUFFER_DAYS = 5
PROCESSING_DELAY = 2  # seconds between date processing

def get_date_range(days_back: int, max_days: int = MAX_FETCH_DAYS, buffer_days: int = BUFFER_DAYS) -> tuple[datetime, datetime, str]:
    """Get consistent date range with constraints."""
    limited_days = min(days_back, max_days)
    end_date = datetime.now().date() - timedelta(days=buffer_days)
    start_date = end_date - timedelta(days=limited_days - 1)
    date_str = start_date.strftime('%Y-%m-%d')
    
    if days_back > max_days:
        logger.warning(f"Requested {days_back} days, limited to {max_days} days")
    if days_back != limited_days:
        logger.info(f"Adjusted to {limited_days} days ending {buffer_days} days ago ({end_date})")
    
    return start_date, end_date, date_str

def create_quality_check_entry(city_name: str, error_msg: str = None, date_str: str = None) -> dict:
    """Create standardized quality check entry for failed processing."""
    if error_msg:
        issue = f"Processing error for {city_name}"
        if date_str:
            issue += f" on {date_str}"
        issue += f": {error_msg}"
    else:
        issue = f"No valid data for {city_name}"
        if date_str:
            issue += f" on {date_str}"
    
    return {
        'passed': False,
        'issues': [issue],
        'summary': {'total_rows': 0, 'missing_values': 0, 'outliers': 0}
    }

def process_city_data_safe(fetcher: DataFetcher, processor: DataProcessor, 
                          city, date_str: str) -> tuple[pd.DataFrame, dict]:
    """Safely process data for a single city on a single date."""
    try:
        weather_data, energy_data = fetcher.fetch_city_data(city, date_str, date_str)
        df = processor.process_city_data(weather_data, energy_data, city)
        
        if df is not None and not df.empty:
            quality_report = processor.check_data_quality(df)
            logger.debug(f"Processed {city.name} for {date_str}: {len(df)} records")
            return df, quality_report
        else:
            logger.warning(f"No valid data for {city.name} on {date_str}")
            return None, create_quality_check_entry(city.name, date_str=date_str)
            
    except Exception as e:
        logger.error(f"Processing failed for {city.name} on {date_str}: {str(e)}")
        return None, create_quality_check_entry(city.name, str(e), date_str)

def save_processed_data(config: Config, combined_df: pd.DataFrame, quality_checks: list,
                       file_prefix: str, date_info: str = "") -> bool:
    """Save processed data and quality reports with consistent naming."""
    if combined_df.empty:
        logger.error("No data to save")
        return False
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = f'{file_prefix}_{date_info}_{timestamp}.csv' if date_info else f'{file_prefix}_{timestamp}.csv'
    file_path = os.path.join(config.data_paths['processed'], filename)
    os.makedirs(config.data_paths['processed'], exist_ok=True)
    combined_df.to_csv(file_path, index=False)
    logger.info(f"Saved data to {file_path}: {len(combined_df)} records")
    
    if file_prefix in ['daily', 'historical']:
        latest_path = os.path.join(config.data_paths['processed'], f'latest_{file_prefix}.csv')
        combined_df.to_csv(latest_path, index=False)
        logger.info(f"Saved latest {file_prefix} data to {latest_path}")
    
    processor = DataProcessor(config)
    quality_summary = processor.generate_quality_summary(quality_checks)
    quality_filename = f'quality_report_{file_prefix}_{timestamp}.json'
    quality_path = os.path.join(config.data_paths['processed'], quality_filename)
    
    with open(quality_path, 'w') as f:
        json.dump(quality_summary, f, indent=2, default=str)
    logger.info(f"Saved quality report to {quality_path}")
    
    return True

def run_pipeline(config: Config, pipeline_type: str, days: int = MAX_FETCH_DAYS) -> bool:
    """Generic pipeline runner for different pipeline types."""
    logger.info(f"Starting {pipeline_type} pipeline for {days} days")
    
    fetcher = DataFetcher(config)
    processor = DataProcessor(config)
    all_data = []
    quality_checks = []
    
    if pipeline_type == "daily":
        _, _, date_str = get_date_range(1)
        logger.info(f"Processing data for {date_str}")
        for city in config.cities:
            logger.info(f"Processing {city.name}")
            df, quality_report = process_city_data_safe(fetcher, processor, city, date_str)
            if df is not None:
                all_data.append(df)
                logger.info(f"Successfully processed data for {city.name}: {len(df)} records")
            quality_checks.append(quality_report)
    
    else:  # recent or historical
        start_date, end_date, _ = get_date_range(days)
        logger.info(f"Fetching data from {start_date} to {end_date}")
        chunk_size = config.rate_limits['chunk_size_days']
        current_start = start_date
        
        while current_start <= end_date:
            chunk_end = min(current_start + timedelta(days=chunk_size - 1), end_date)
            chunk_start_str = current_start.strftime('%Y-%m-%d')
            chunk_end_str = chunk_end.strftime('%Y-%m-%d')
            logger.info(f"Processing chunk: {chunk_start_str} to {chunk_end_str}")
            
            for city in config.cities:
                try:
                    weather_data, energy_data = fetcher.fetch_city_data(city, chunk_start_str, chunk_end_str)
                    df = processor.process_city_data(weather_data, energy_data, city)
                    if df is not None and not df.empty:
                        quality_report = processor.check_data_quality(df)
                        all_data.append(df)
                        quality_checks.append(quality_report)
                        logger.info(f"Processed chunk for {city.name}: {len(df)} records")
                    else:
                        logger.warning(f"No valid data for {city.name} in chunk {chunk_start_str}-{chunk_end_str}")
                        quality_checks.append(create_quality_check_entry(city.name, date_str=chunk_start_str))
                except Exception as e:
                    logger.error(f"Processing failed for {city.name} in chunk {chunk_start_str}-{chunk_end_str}: {str(e)}")
                    quality_checks.append(create_quality_check_entry(city.name, str(e), chunk_start_str))
            
            current_start = chunk_end + timedelta(days=1)
            if current_start <= end_date:
                time.sleep(PROCESSING_DELAY)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        actual_days = min(days, MAX_FETCH_DAYS)
        start_range = (datetime.now().date() - timedelta(days=actual_days + BUFFER_DAYS - 1)).strftime('%Y-%m-%d')
        end_range = (datetime.now().date() - timedelta(days=BUFFER_DAYS)).strftime('%Y-%m-%d')
        date_info = f'{actual_days}days_{start_range}_to_{end_range}' if pipeline_type != 'daily' else date_str
        return save_processed_data(config, combined_df, quality_checks, pipeline_type, date_info)
    else:
        logger.error(f"No {pipeline_type} data was successfully processed")
        save_processed_data(config, pd.DataFrame(), quality_checks, pipeline_type)
        return False

def run_pipeline_with_validation(config: Config, pipeline_type: str = "daily", days: int = MAX_FETCH_DAYS) -> bool:
    """Run pipeline with comprehensive validation and error handling."""
    start_time = datetime.now()
    logger.info(f"Starting {pipeline_type} pipeline at {start_time}")
    
    try:
        errors = config.validate()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        for path_key, path_value in config.data_paths.items():
            os.makedirs(path_value, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path_value}")
        
        return run_pipeline(config, pipeline_type, days)
    
    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"{pipeline_type.title()} pipeline failed after {duration}: {str(e)}")
        return False

if __name__ == "__main__":
    log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    os.makedirs('logs', exist_ok=True)
    logger.add(log_file, rotation="10 MB", level="INFO")
    logger.info("Pipeline script started")
    
    try:
        config = Config.load()
        
        historical_success = run_pipeline_with_validation(config, "historical", days=MAX_FETCH_DAYS)
        daily_success = run_pipeline_with_validation(config, "daily")
        
        if historical_success:
            logger.info("Historical pipeline succeeded")
        else:
            logger.error("Historical pipeline failed")
        
        if daily_success:
            logger.info("Daily pipeline succeeded")
        
        if historical_success and daily_success:
            logger.info("All pipelines completed successfully")
        else:
            logger.error("Pipeline execution had issues")
            exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline script failed with critical error: {str(e)}")
        exit(1)
    finally:
        logger.info("Pipeline script finished")