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

def get_date_range(config: Config, days_back: int) -> tuple[datetime, datetime, str]:
    """Get consistent date range with constraints from config."""
    max_days = config.rate_limits['max_fetch_days']
    buffer_days = config.rate_limits['buffer_days']
    
    limited_days = min(days_back, max_days)
    today = datetime.now().date()
    end_date = today - timedelta(days=buffer_days)
    start_date = end_date - timedelta(days=limited_days - 1)
    
    # For daily pipeline (days_back=1), we want just the end_date
    if days_back == 1:
        date_str = end_date.strftime('%Y-%m-%d')
    else:
        date_str = start_date.strftime('%Y-%m-%d')
    
    if days_back > max_days:
        logger.warning(f"Requested {days_back} days, limited to {max_days} days")
    if days_back != limited_days:
        logger.info(f"Adjusted to {limited_days} days ending {buffer_days} days ago ({end_date})")
    
    logger.info(f"Date range calculation: Today={today}, Buffer={buffer_days} days, End date={end_date}, Start date={start_date}")
    
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
        'summary': {'total_rows': 0, 'missing_values': 0, 'outliers': 0},
        'city': city_name,
        'date': date_str,
        'timestamp': datetime.now().isoformat()
    }

def process_city_data_safe(fetcher: DataFetcher, processor: DataProcessor, 
                          city, date_str: str) -> tuple[pd.DataFrame, dict]:
    """Safely process data for a single city on a single date."""
    try:
        logger.debug(f"Fetching data for {city.name} on {date_str}")
        weather_data, energy_data = fetcher.fetch_city_data(city, date_str, date_str)
        
        # Check if we have any data before processing
        if not weather_data and not energy_data:
            logger.warning(f"No weather or energy data available for {city.name} on {date_str}")
            return None, create_quality_check_entry(city.name, "No data from APIs", date_str)
        
        # Log what data we received
        weather_count = len(weather_data) if weather_data else 0
        energy_count = len(energy_data) if energy_data else 0
        logger.debug(f"Retrieved for {city.name}: {weather_count} weather records, {energy_count} energy records")
        
        df = processor.process_city_data(weather_data, energy_data, city)
        
        if df is not None and not df.empty:
            quality_report = processor.check_data_quality(df)
            logger.info(f"Successfully processed {city.name} for {date_str}: {len(df)} records")
            return df, quality_report
        else:
            logger.warning(f"Processing returned empty dataset for {city.name} on {date_str}")
            return None, create_quality_check_entry(city.name, "Empty dataset after processing", date_str)
            
    except Exception as e:
        logger.error(f"Processing failed for {city.name} on {date_str}: {str(e)}")
        return None, create_quality_check_entry(city.name, str(e), date_str)

def save_processed_data(config: Config, combined_df: pd.DataFrame, quality_checks: list,
                       file_prefix: str, date_info: str = "", allow_empty: bool = False) -> bool:
    """Save processed data and quality reports with consistent naming."""
    
    # Always save quality reports, even if no data
    processor = DataProcessor(config)
    quality_summary = processor.generate_quality_summary(quality_checks)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    quality_filename = f'quality_report_{file_prefix}_{date_info}_{timestamp}.json' if date_info else f'quality_report_{file_prefix}_{timestamp}.json'
    quality_path = os.path.join(config.data_paths['processed'], quality_filename)
    
    os.makedirs(config.data_paths['processed'], exist_ok=True)
    
    with open(quality_path, 'w') as f:
        json.dump(quality_summary, f, indent=2, default=str)
    logger.info(f"Saved quality report to {quality_path}")
    
    if combined_df.empty:
        if allow_empty:
            logger.warning("No data to save, but continuing due to allow_empty=True")
            return True
        else:
            logger.error("No data to save")
            return False
    
    filename = f'{file_prefix}_{date_info}_{timestamp}.csv' if date_info else f'{file_prefix}_{timestamp}.csv'
    file_path = os.path.join(config.data_paths['processed'], filename)
    combined_df.to_csv(file_path, index=False)
    logger.info(f"Saved data to {file_path}: {len(combined_df)} records")
    
    if file_prefix in ['daily', 'historical']:
        latest_path = os.path.join(config.data_paths['processed'], f'latest_{file_prefix}.csv')
        combined_df.to_csv(latest_path, index=False)
        logger.info(f"Saved latest {file_prefix} data to {latest_path}")
    
    return True

def run_pipeline(config: Config, pipeline_type: str, days: int = None) -> bool:
    """Generic pipeline runner for different pipeline types."""
    # Use config default if days not specified
    if days is None:
        days = config.rate_limits['max_fetch_days']
    
    logger.info(f"Starting {pipeline_type} pipeline for {days} days")
    
    fetcher = DataFetcher(config)
    processor = DataProcessor(config)
    all_data = []
    quality_checks = []
    successful_cities = 0
    total_cities = len(config.cities)
    
    # Get processing delay from config
    processing_delay = config.rate_limits.get('processing_delay_seconds', 2)
    
    if pipeline_type == "daily":
        _, end_date, date_str = get_date_range(config, 1)  # Get the actual end_date for daily processing
        buffer_days = config.rate_limits['buffer_days']
        logger.info(f"Processing daily data for {date_str} (with {buffer_days}-day buffer from today)")
        
        for city in config.cities:
            logger.info(f"Processing {city.name}")
            df, quality_report = process_city_data_safe(fetcher, processor, city, date_str)
            
            if df is not None and not df.empty:
                all_data.append(df)
                successful_cities += 1
                logger.info(f"Successfully processed data for {city.name}: {len(df)} records")
            else:
                logger.warning(f"No valid data for {city.name} on {date_str}")
            
            quality_checks.append(quality_report)
    
    else:  # recent or historical
        start_date, end_date, _ = get_date_range(config, days)
        logger.info(f"Processing {pipeline_type} data from {start_date} to {end_date} across {total_cities} cities")
        chunk_size = config.rate_limits['chunk_size_days']
        current_start = start_date
        
        while current_start <= end_date:
            chunk_end = min(current_start + timedelta(days=chunk_size - 1), end_date)
            chunk_start_str = current_start.strftime('%Y-%m-%d')
            chunk_end_str = chunk_end.strftime('%Y-%m-%d')
            logger.info(f"Processing chunk: {chunk_start_str} to {chunk_end_str}")
            
            for city in config.cities:
                try:
                    logger.debug(f"Fetching chunk data for {city.name}")
                    weather_data, energy_data = fetcher.fetch_city_data(city, chunk_start_str, chunk_end_str)
                    
                    # Check if we have any data
                    weather_count = len(weather_data) if weather_data else 0
                    energy_count = len(energy_data) if energy_data else 0
                    
                    if weather_count == 0 and energy_count == 0:
                        logger.warning(f"No data from APIs for {city.name} in chunk {chunk_start_str}-{chunk_end_str}")
                        quality_checks.append(create_quality_check_entry(city.name, "No data from APIs", chunk_start_str))
                        continue
                    
                    logger.debug(f"Retrieved for {city.name}: {weather_count} weather records, {energy_count} energy records")
                    df = processor.process_city_data(weather_data, energy_data, city)
                    
                    if df is not None and not df.empty:
                        quality_report = processor.check_data_quality(df)
                        all_data.append(df)
                        quality_checks.append(quality_report)
                        successful_cities += 1
                        logger.info(f"Processed chunk for {city.name}: {len(df)} records")
                    else:
                        logger.warning(f"Processing returned empty dataset for {city.name} in chunk {chunk_start_str}-{chunk_end_str}")
                        quality_checks.append(create_quality_check_entry(city.name, "Empty dataset after processing", chunk_start_str))
                        
                except Exception as e:
                    logger.error(f"Processing failed for {city.name} in chunk {chunk_start_str}-{chunk_end_str}: {str(e)}")
                    quality_checks.append(create_quality_check_entry(city.name, str(e), chunk_start_str))
            
            current_start = chunk_end + timedelta(days=1)
            if current_start <= end_date:
                time.sleep(processing_delay)
    
    # Log summary of processing results
    logger.info(f"Processing summary: {successful_cities}/{total_cities} cities processed successfully")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        max_days = config.rate_limits['max_fetch_days']
        buffer_days = config.rate_limits['buffer_days']
        actual_days = min(days, max_days)
        
        # Updated to use buffer_days from config consistently
        start_range = (datetime.now().date() - timedelta(days=actual_days + buffer_days - 1)).strftime('%Y-%m-%d')
        end_range = (datetime.now().date() - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        date_info = f'{actual_days}days_{start_range}_to_{end_range}' if pipeline_type != 'daily' else date_str
        
        success = save_processed_data(config, combined_df, quality_checks, pipeline_type, date_info)
        if success:
            logger.info(f"{pipeline_type.title()} pipeline completed with data from {successful_cities} cities")
        return success
    else:
        logger.warning(f"No {pipeline_type} data was successfully processed from any city")
        # Still save quality reports even with no data
        max_days = config.rate_limits['max_fetch_days']
        buffer_days = config.rate_limits['buffer_days']
        actual_days = min(days, max_days)
        start_range = (datetime.now().date() - timedelta(days=actual_days + buffer_days - 1)).strftime('%Y-%m-%d')
        end_range = (datetime.now().date() - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        date_info = f'{actual_days}days_{start_range}_to_{end_range}' if pipeline_type != 'daily' else get_date_range(config, 1)[2]
        
        save_processed_data(config, pd.DataFrame(), quality_checks, pipeline_type, date_info, allow_empty=True)
        
        # For daily pipeline, this might be expected if data isn't available yet
        if pipeline_type == "daily":
            logger.warning("Daily pipeline completed with no data - this may be normal if data isn't available yet")
            return True  # Consider this a success for daily pipeline
        else:
            return False

def run_pipeline_with_validation(config: Config, pipeline_type: str = "daily", days: int = None) -> bool:
    """Run pipeline with comprehensive validation and error handling."""
    start_time = datetime.now()
    logger.info(f"Starting {pipeline_type} pipeline at {start_time}")
    
    try:
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        # Ensure directories exist
        for path_key, path_value in config.data_paths.items():
            os.makedirs(path_value, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path_value}")
        
        # Run the pipeline
        result = run_pipeline(config, pipeline_type, days)
        
        duration = datetime.now() - start_time
        if result:
            logger.info(f"{pipeline_type.title()} pipeline completed successfully in {duration}")
        else:
            logger.warning(f"{pipeline_type.title()} pipeline completed with issues in {duration}")
        
        return result
    
    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"{pipeline_type.title()} pipeline failed after {duration}: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Load config first to get logging configuration
        config = Config.load()
        
        # Setup logging based on config
        log_file = os.path.join(config.data_paths['logs'], f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
        os.makedirs(config.data_paths['logs'], exist_ok=True)
        
        logger.add(
            log_file, 
            rotation=config.logging.get('rotation', '10 MB'), 
            level=config.logging.get('level', 'INFO'),
            retention=config.logging.get('retention', '7 days')
        )
        logger.info("Pipeline script started")
        
        # Get max_fetch_days from config
        max_fetch_days = config.rate_limits['max_fetch_days']
        
        # Run historical pipeline first
        logger.info("=" * 50)
        logger.info("STARTING HISTORICAL PIPELINE")
        logger.info("=" * 50)
        historical_success = run_pipeline_with_validation(config, "historical", days=max_fetch_days)
        
        # Run daily pipeline
        logger.info("=" * 50)
        logger.info("STARTING DAILY PIPELINE")
        logger.info("=" * 50)
        daily_success = run_pipeline_with_validation(config, "daily")
        
        # Final summary
        logger.info("=" * 50)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        
        if historical_success:
            logger.info("✓ Historical pipeline succeeded")
        else:
            logger.error("✗ Historical pipeline failed")
        
        if daily_success:
            logger.info("✓ Daily pipeline succeeded")
        else:
            logger.warning("⚠ Daily pipeline had issues (may be normal if data not available)")
        
        # Overall success determination
        if historical_success and daily_success:
            logger.info("🎉 All pipelines completed successfully")
            exit(0)
        elif historical_success:
            logger.warning("⚠ Pipelines completed with some issues (historical succeeded)")
            exit(0)  # Don't fail if only daily has issues
        else:
            logger.error("❌ Pipeline execution failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline script failed with critical error: {str(e)}")
        exit(1)
    finally:
        logger.info("Pipeline script finished")