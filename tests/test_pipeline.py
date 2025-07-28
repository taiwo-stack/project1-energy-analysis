"""
Test suite for pipeline.py module.

This module tests the pipeline orchestration logic including:
- Date range calculation
- Quality check entry creation
- Safe city data processing
- Data and quality report saving
- Pipeline execution for daily and historical modes
- Pipeline validation and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, mock_open, call
import os
import json
from src.pipeline import (
    get_date_range,
    create_quality_check_entry,
    process_city_data_safe,
    save_processed_data,
    run_pipeline,
    run_pipeline_with_validation
)
from config import Config, City
from src.data_fetcher import DataFetcher
from src.data_processor import DataProcessor

class TestPipeline:
    """Test suite for the pipeline module."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        cities = [
            City(name="New York", state="NY", noaa_station="NYC123", eia_region="NYISO", lat=40.7128, lon=-74.0060),
            City(name="Chicago", state="IL", noaa_station="CHI456", eia_region="PJM", lat=41.8781, lon=-87.6298),
        ]
        config = Mock(spec=Config)
        config.cities = cities
        config.data_paths = {'processed': 'data/processed'}
        config.rate_limits = {'chunk_size_days': 30}
        config.quality_checks = {
            'temperature': {'max_fahrenheit': 130, 'min_fahrenheit': -50},
            'energy': {'min_value': 0},
            'freshness': {'max_age_hours': 48},
            'completeness': {'min_coverage': 0.8}
        }
        config.validate = Mock(return_value=[])
        config.get_city_by_name = Mock(side_effect=lambda name: next(
            (city for city in cities if city.name.lower() == name.lower()), None
        ))
        return config

    @pytest.fixture
    def mock_fetcher(self):
        """Create a mock DataFetcher for testing."""
        fetcher = Mock(spec=DataFetcher)
        return fetcher

    @pytest.fixture
    def mock_processor(self, mock_config):
        """Create a real DataProcessor for testing, using the fixed implementation."""
        return DataProcessor(mock_config)

    @pytest.fixture
    def sample_noaa_data(self):
        """Create sample NOAA weather data."""
        end_date = datetime.now().date() - timedelta(days=2)
        return {
            'results': [
                {'date': str(end_date), 'datatype': 'TMAX', 'value': 750},
                {'date': str(end_date), 'datatype': 'TMIN', 'value': 550},
                {'date': str(end_date + timedelta(days=1)), 'datatype': 'TMAX', 'value': 780},
                {'date': str(end_date + timedelta(days=1)), 'datatype': 'TMIN', 'value': 580}
            ]
        }

    @pytest.fixture
    def sample_eia_data(self):
        """Create sample EIA energy data."""
        end_date = datetime.now().date() - timedelta(days=2)
        return {
            'response': {
                'data': [
                    {'period': str(end_date), 'value': 1000, 'respondent': 'NYISO'},
                    {'period': str(end_date + timedelta(days=1)), 'value': 1050, 'respondent': 'NYISO'}
                ]
            }
        }

    @pytest.fixture
    def sample_processed_df(self):
        """Create sample processed DataFrame."""
        end_date = datetime.now().date() - timedelta(days=2)
        return pd.DataFrame({
            'date': pd.to_datetime([end_date, end_date + timedelta(days=1)]),
            'station_id': ['NYC123', 'NYC123'],
            'temperature_max': [75.0, 78.0],
            'temperature_min': [55.0, 58.0],
            'temperature_avg': [65.0, 68.0],
            'data_source': ['NOAA', 'NOAA'],
            'processed_at': [datetime.now()] * 2,
            'energy_demand': [1000, 1050],
            'eia_region': ['NYISO', 'NYISO'],
            'city': ['New York', 'New York'],
            'day_of_week': ['Saturday', 'Sunday'],
            'is_weekend': [True, True]
        })

    @pytest.fixture
    def sample_quality_report(self):
        """Create sample quality report."""
        return {
            'passed': True,
            'issues': [],
            'summary': {'total_rows': 2, 'missing_values': 0, 'outliers': 0, 'data_freshness_hours': 24}
        }

    @pytest.mark.parametrize("days,expected_days_back", [
        (1, 2),    # Daily: 2 days back from current date
        (10, (10 + 1)),  # Historical: days + 1 back from current date  
        (100, 91)  # Max days exceeded: MAX_FETCH_DAYS + 1 back
    ])
    def test_get_date_range_parametrized(self, days, expected_days_back):
        """Test get_date_range with different day values using parametrization."""
        with patch('src.pipeline.datetime') as mock_datetime:
            test_date = datetime(2025, 7, 28)
            mock_datetime.now.return_value = test_date
            
            start_date, end_date, date_str = get_date_range(days)
            
            expected_end = test_date.date() - timedelta(days=2)
            expected_start = test_date.date() - timedelta(days=expected_days_back)
            
            assert end_date == expected_end
            assert start_date == expected_start
            assert date_str == str(expected_start)

    def test_get_date_range_daily(self):
        """Test get_date_range for daily pipeline."""
        with patch('src.pipeline.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            start_date, end_date, date_str = get_date_range(1)
            assert start_date == datetime(2025, 7, 26).date()
            assert end_date == datetime(2025, 7, 26).date()
            assert date_str == '2025-07-26'

    def test_get_date_range_historical(self):
        """Test get_date_range for historical pipeline."""
        with patch('src.pipeline.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            start_date, end_date, date_str = get_date_range(10)
            assert start_date == datetime(2025, 7, 17).date()
            assert end_date == datetime(2025, 7, 26).date()
            assert date_str == '2025-07-17'

    def test_get_date_range_max_days(self):
        """Test get_date_range with days exceeding MAX_FETCH_DAYS."""
        with patch('src.pipeline.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            start_date, end_date, date_str = get_date_range(100)
            assert start_date == datetime(2025, 4, 28).date()  # 90 days back from 2025-07-26
            assert end_date == datetime(2025, 7, 26).date()
            assert date_str == '2025-04-28'

    def test_create_quality_check_entry_no_error(self):
        """Test create_quality_check_entry with no error message."""
        city_name = "New York"
        date_str = "2025-07-26"
        
        with patch('src.pipeline.datetime') as mock_datetime:
            test_time = datetime(2025, 7, 28, 10, 30, 45)
            mock_datetime.now.return_value = test_time
            
            result = create_quality_check_entry(city_name, date_str=date_str)
            
            assert result['passed'] is False
            assert result['issues'] == [f"No valid data for {city_name} on {date_str}"]
            assert result['summary'] == {'total_rows': 0, 'missing_values': 0, 'outliers': 0}
            assert result['city'] == city_name
            assert result['date'] == date_str
            assert result['timestamp'] == test_time.isoformat()

    def test_create_quality_check_entry_with_error(self):
        """Test create_quality_check_entry with an error message."""
        city_name = "New York"
        error_msg = "API timeout"
        date_str = "2025-07-26"
        
        with patch('src.pipeline.datetime') as mock_datetime:
            test_time = datetime(2025, 7, 28, 10, 30, 45)
            mock_datetime.now.return_value = test_time
            
            result = create_quality_check_entry(city_name, error_msg, date_str)
            
            assert result['passed'] is False
            assert result['issues'] == [f"Processing error for {city_name} on {date_str}: {error_msg}"]
            assert result['summary'] == {'total_rows': 0, 'missing_values': 0, 'outliers': 0}
            assert result['city'] == city_name
            assert result['date'] == date_str
            assert result['timestamp'] == test_time.isoformat()

    def test_process_city_data_safe_success(self, mock_fetcher, mock_processor, sample_noaa_data, sample_eia_data, sample_processed_df, sample_quality_report):
        """Test process_city_data_safe with successful data processing."""
        city = City(name="New York", state="NY", noaa_station="NYC123", eia_region="NYISO", lat=40.7128, lon=-74.0060)
        date_str = "2025-07-26"
        mock_fetcher.fetch_city_data.return_value = (sample_noaa_data, sample_eia_data)
        
        # Mock the processor's process_data method to return expected results
        mock_processor.process_data.return_value = (sample_processed_df, sample_quality_report)
        
        df, quality_report = process_city_data_safe(mock_fetcher, mock_processor, city, date_str)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        pd.testing.assert_frame_equal(df, sample_processed_df, check_dtype=False)
        assert quality_report['passed'] is True
        assert quality_report['issues'] == []
        assert quality_report['summary']['total_rows'] == 2
        mock_fetcher.fetch_city_data.assert_called_once_with(city, date_str, date_str)

    def test_process_city_data_safe_no_data(self, mock_fetcher, mock_processor):
        """Test process_city_data_safe with no data from APIs."""
        city = City(name="New York", state="NY", noaa_station="NYC123", eia_region="NYISO", lat=40.7128, lon=-74.0060)
        date_str = "2025-07-26"
        mock_fetcher.fetch_city_data.return_value = (None, None)

        df, quality_report = process_city_data_safe(mock_fetcher, mock_processor, city, date_str)
        
        assert df is None
        assert quality_report['passed'] is False
        assert quality_report['issues'] == [f"Processing error for New York on {date_str}: No data from APIs"]
        assert quality_report['summary'] == {'total_rows': 0, 'missing_values': 0, 'outliers': 0}
        mock_fetcher.fetch_city_data.assert_called_once_with(city, date_str, date_str)

    def test_process_city_data_safe_error_handling(self, mock_fetcher, mock_processor):
        """Test process_city_data_safe with different types of exceptions."""
        city = City(name="New York", state="NY", noaa_station="NYC123", eia_region="NYISO", lat=40.7128, lon=-74.0060)
        date_str = "2025-07-26"
        
        # Test different exception types
        test_cases = [
            (ConnectionError("Network error"), "Network error"),
            (ValueError("Invalid data format"), "Invalid data format"),
            (KeyError("Missing key"), "Missing key"),
            (Exception("Generic error"), "Generic error")
        ]
        
        for exception, expected_msg in test_cases:
            mock_fetcher.fetch_city_data.side_effect = exception
            
            df, quality_report = process_city_data_safe(mock_fetcher, mock_processor, city, date_str)
            
            assert df is None
            assert quality_report['passed'] is False
            assert quality_report['issues'] == [f"Processing error for New York on {date_str}: {expected_msg}"]
            assert quality_report['summary'] == {'total_rows': 0, 'missing_values': 0, 'outliers': 0}

    def test_save_processed_data_success(self, mock_config, sample_processed_df, sample_quality_report):
        """Test save_processed_data with valid data."""
        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump, \
             patch.object(sample_processed_df, 'to_csv') as mock_to_csv:
            
            result = save_processed_data(mock_config, sample_processed_df, [sample_quality_report], "daily", "2025-07-26")
            
            assert result is True
            mock_makedirs.assert_called_once_with(mock_config.data_paths['processed'], exist_ok=True)
            mock_json_dump.assert_called_once()
            
            # Verify CSV files are saved with correct names
            assert mock_to_csv.call_count == 2
            csv_calls = mock_to_csv.call_args_list
            assert any('daily_2025-07-26' in str(call) for call in csv_calls)
            assert any('daily_latest' in str(call) for call in csv_calls)

    @pytest.mark.parametrize("allow_empty,expected_result", [
        (False, False),
        (True, True)
    ])
    def test_save_processed_data_empty_dataframe(self, mock_config, sample_quality_report, allow_empty, expected_result):
        """Test save_processed_data with empty DataFrame using parametrization."""
        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            
            result = save_processed_data(
                mock_config, pd.DataFrame(), [sample_quality_report], 
                "daily", "2025-07-26", allow_empty=allow_empty
            )
            
            assert result is expected_result
            mock_makedirs.assert_called_once_with(mock_config.data_paths['processed'], exist_ok=True)
            mock_json_dump.assert_called_once()
            mock_file.assert_called_once()  # Only quality report is saved

    def test_save_processed_data_io_error(self, mock_config, sample_processed_df, sample_quality_report):
        """Test save_processed_data handles IO errors gracefully."""
        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump', side_effect=IOError("Disk full")):
            
            result = save_processed_data(mock_config, sample_processed_df, [sample_quality_report], "daily", "2025-07-26")
            
            # Should handle the error and return False
            assert result is False

    def test_run_pipeline_daily_success(self, mock_config, mock_fetcher, mock_processor, sample_processed_df, sample_quality_report):
        """Test run_pipeline in daily mode with successful processing."""
        with patch('src.pipeline.datetime') as mock_datetime, \
             patch('src.pipeline.get_date_range') as mock_get_date_range, \
             patch('src.pipeline.process_city_data_safe') as mock_process_city_data_safe, \
             patch('src.pipeline.save_processed_data') as mock_save_processed_data:
            
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            mock_get_date_range.return_value = (datetime(2025, 7, 26).date(), datetime(2025, 7, 26).date(), '2025-07-26')
            mock_process_city_data_safe.side_effect = [
                (sample_processed_df, sample_quality_report),  # New York
                (None, create_quality_check_entry("Chicago", "No data from APIs", "2025-07-26"))  # Chicago fails
            ]
            mock_save_processed_data.return_value = True

            result = run_pipeline(mock_config, "daily")
            
            assert result is True
            mock_get_date_range.assert_called_once_with(1)
            assert mock_process_city_data_safe.call_count == 2
            mock_save_processed_data.assert_called_once()
            
            # Verify the correct data is passed to save function
            save_call_args = mock_save_processed_data.call_args[0]
            assert save_call_args[1].equals(sample_processed_df)  # Only New York's data
            assert len(save_call_args[2]) == 2  # Quality checks for both cities

    def test_run_pipeline_daily_all_cities_fail(self, mock_config, mock_fetcher, mock_processor):
        """Test run_pipeline in daily mode when all cities fail to provide data."""
        with patch('src.pipeline.datetime') as mock_datetime, \
             patch('src.pipeline.get_date_range') as mock_get_date_range, \
             patch('src.pipeline.process_city_data_safe') as mock_process_city_data_safe, \
             patch('src.pipeline.save_processed_data') as mock_save_processed_data:
            
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            mock_get_date_range.return_value = (datetime(2025, 7, 26).date(), datetime(2025, 7, 26).date(), '2025-07-26')
            mock_process_city_data_safe.return_value = (None, create_quality_check_entry("Test City", "No data from APIs", "2025-07-26"))
            mock_save_processed_data.return_value = True

            result = run_pipeline(mock_config, "daily")
            
            assert result is True  # Daily mode should still succeed even with no data
            assert mock_process_city_data_safe.call_count == 2
            mock_save_processed_data.assert_called_once_with(
                mock_config, mock.ANY, mock.ANY, "daily", '2025-07-26', allow_empty=True
            )

    def test_run_pipeline_historical_chunking(self, mock_config, mock_fetcher, mock_processor, sample_processed_df, sample_quality_report):
        """Test run_pipeline historical mode with date chunking."""
        with patch('src.pipeline.datetime') as mock_datetime, \
             patch('src.pipeline.get_date_range') as mock_get_date_range, \
             patch('src.pipeline.process_city_data_safe') as mock_process_city_data_safe, \
             patch('src.pipeline.save_processed_data') as mock_save_processed_data, \
             patch('time.sleep') as mock_sleep:
            
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            mock_get_date_range.return_value = (datetime(2025, 6, 28).date(), datetime(2025, 7, 26).date(), '2025-06-28')  # 28 days
            mock_process_city_data_safe.return_value = (sample_processed_df, sample_quality_report)
            mock_save_processed_data.return_value = True

            result = run_pipeline(mock_config, "historical", days=30)
            
            assert result is True
            mock_get_date_range.assert_called_once_with(30)
            # Should process in chunks, so multiple calls expected
            assert mock_process_city_data_safe.call_count >= 2
            mock_save_processed_data.assert_called_once()

    def test_run_pipeline_historical_no_data_fails(self, mock_config, mock_fetcher, mock_processor):
        """Test run_pipeline in historical mode fails when no data is collected."""
        with patch('src.pipeline.datetime') as mock_datetime, \
             patch('src.pipeline.get_date_range') as mock_get_date_range, \
             patch('src.pipeline.process_city_data_safe') as mock_process_city_data_safe, \
             patch('src.pipeline.save_processed_data') as mock_save_processed_data, \
             patch('time.sleep') as mock_sleep:
            
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            mock_get_date_range.return_value = (datetime(2025, 7, 17).date(), datetime(2025, 7, 26).date(), '2025-07-17')
            mock_process_city_data_safe.return_value = (None, create_quality_check_entry("Test City", "No data from APIs", "2025-07-17"))
            mock_save_processed_data.return_value = True

            result = run_pipeline(mock_config, "historical", days=10)
            
            assert result is False  # Historical mode should fail when no data is collected
            mock_save_processed_data.assert_called_once_with(
                mock_config, mock.ANY, mock.ANY, "historical", mock.ANY, allow_empty=True
            )

    def test_run_pipeline_with_validation_success(self, mock_config):
        """Test run_pipeline_with_validation with successful execution."""
        with patch('src.pipeline.datetime') as mock_datetime, \
             patch('src.pipeline.run_pipeline') as mock_run_pipeline, \
             patch('os.makedirs') as mock_makedirs:
            
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            mock_run_pipeline.return_value = True
            mock_config.validate.return_value = []

            result = run_pipeline_with_validation(mock_config, "daily")
            
            assert result is True
            mock_makedirs.assert_called()
            mock_run_pipeline.assert_called_once_with(mock_config, "daily", 90)  # Default days

    def test_run_pipeline_with_validation_custom_days(self, mock_config):
        """Test run_pipeline_with_validation with custom days parameter."""
        with patch('src.pipeline.datetime') as mock_datetime, \
             patch('src.pipeline.run_pipeline') as mock_run_pipeline, \
             patch('os.makedirs') as mock_makedirs:
            
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            mock_run_pipeline.return_value = True
            mock_config.validate.return_value = []

            result = run_pipeline_with_validation(mock_config, "historical", days=30)
            
            assert result is True
            mock_run_pipeline.assert_called_once_with(mock_config, "historical", 30)

    def test_run_pipeline_with_validation_config_errors(self, mock_config):
        """Test run_pipeline_with_validation with multiple configuration errors."""
        mock_config.validate.return_value = ["Invalid API key", "Missing data path", "Invalid city configuration"]
        
        with patch('src.pipeline.run_pipeline') as mock_run_pipeline:
            result = run_pipeline_with_validation(mock_config, "daily")
            
            assert result is False
            mock_run_pipeline.assert_not_called()

    def test_run_pipeline_with_validation_exception_handling(self, mock_config):
        """Test run_pipeline_with_validation handles different types of exceptions."""
        exceptions_to_test = [
            ConnectionError("Network unavailable"),
            FileNotFoundError("Config file missing"),
            PermissionError("Access denied"),
            Exception("Unexpected error")
        ]
        
        for exception in exceptions_to_test:
            with patch('src.pipeline.datetime') as mock_datetime, \
                 patch('src.pipeline.run_pipeline') as mock_run_pipeline:
                
                mock_datetime.now.return_value = datetime(2025, 7, 28)
                mock_run_pipeline.side_effect = exception
                mock_config.validate.return_value = []

                result = run_pipeline_with_validation(mock_config, "daily")
                assert result is False

    def test_run_pipeline_with_validation_directory_creation(self, mock_config):
        """Test that run_pipeline_with_validation creates necessary directories."""
        with patch('src.pipeline.datetime') as mock_datetime, \
             patch('src.pipeline.run_pipeline') as mock_run_pipeline, \
             patch('os.makedirs') as mock_makedirs:
            
            mock_datetime.now.return_value = datetime(2025, 7, 28)
            mock_run_pipeline.return_value = True
            mock_config.validate.return_value = []
            mock_config.data_paths = {'processed': 'data/processed', 'logs': 'logs'}

            run_pipeline_with_validation(mock_config, "daily")
            
            # Should create directories for all data paths
            expected_calls = [call(path, exist_ok=True) for path in mock_config.data_paths.values()]
            mock_makedirs.assert_has_calls(expected_calls, any_order=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])