"""
Test suite for data_processor.py module.

This module tests the DataProcessor class functionality including:
- Celsius to Fahrenheit conversion
- NOAA weather data processing
- EIA energy data processing
- City data combination
- Data quality checks
- Quality summary generation
- Data consistency validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.data_processor import DataProcessor
from config import Config, City


class TestDataProcessor:
    """Test suite for the DataProcessor class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        cities = [
            City(name="New York", state="NY", noaa_station="NYC123", eia_region="NYISO", lat=40.7128, lon=-74.0060),
            City(name="Chicago", state="IL", noaa_station="CHI456", eia_region="PJM", lat=41.8781, lon=-87.6298),
        ]
        config = Mock(spec=Config)
        config.cities = cities
        config.quality_checks = {
            'temperature': {'max_fahrenheit': 130, 'min_fahrenheit': -50},
            'energy': {'min_value': 0},
            'freshness': {'max_age_hours': 48},
            'completeness': {'min_coverage': 0.8}
        }
        config.get_city_by_name = Mock(side_effect=lambda name: next(
            (city for city in cities if city.name.lower() == name.lower()), None
        ))
        return config

    @pytest.fixture
    def data_processor(self, mock_config):
        """Create a DataProcessor instance with mock configuration."""
        return DataProcessor(mock_config)

    @pytest.fixture
    def sample_noaa_data(self):
        """Create sample NOAA weather data for testing."""
        return {
            'results': [
                {'date': '2025-07-01', 'datatype': 'TMAX', 'value': 750},  # 75.0°F
                {'date': '2025-07-01', 'datatype': 'TMIN', 'value': 550},  # 55.0°F
                {'date': '2025-07-02', 'datatype': 'TMAX', 'value': 780},  # 78.0°F
                {'date': '2025-07-02', 'datatype': 'TMIN', 'value': 580}   # 58.0°F
            ]
        }

    @pytest.fixture
    def sample_eia_data(self):
        """Create sample EIA energy data for testing."""
        return {
            'response': {
                'data': [
                    {'period': '2025-07-01', 'value': 1000, 'respondent': 'NYISO'},
                    {'period': '2025-07-02', 'value': 1050, 'respondent': 'NYISO'}
                ]
            }
        }

    def test_celsius_to_fahrenheit(self, data_processor):
        """Test Celsius to Fahrenheit conversion."""
        assert data_processor.celsius_to_fahrenheit(0) == 32.0
        assert data_processor.celsius_to_fahrenheit(100) == 212.0
        assert data_processor.celsius_to_fahrenheit(-40) == -40.0
        assert abs(data_processor.celsius_to_fahrenheit(20) - 68.0) < 0.01

    def test_process_noaa_data_valid(self, data_processor, sample_noaa_data):
        """Test processing valid NOAA weather data."""
        result = data_processor.process_noaa_data(sample_noaa_data, "NYC123")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['date', 'station_id', 'temperature_max', 'temperature_min', 'temperature_avg', 'data_source', 'processed_at']
        assert result['temperature_max'].iloc[0] == 750
        assert result['temperature_min'].iloc[0] == 550
        assert result['temperature_avg'].iloc[0] == 650  # (750 + 550) / 2
        assert result['data_source'].iloc[0] == 'NOAA'
        assert (pd.to_datetime(result['date']) == pd.to_datetime(['2025-07-01', '2025-07-02'])).all()

    def test_process_noaa_data_empty(self, data_processor):
        """Test processing empty NOAA data."""
        result = data_processor.process_noaa_data({}, "NYC123")
        assert result is None
        
        result = data_processor.process_noaa_data({'results': []}, "NYC123")
        assert result is None

    def test_process_noaa_data_missing_datatypes(self, data_processor):
        """Test processing NOAA data with missing TMAX/TMIN."""
        raw_data = {
            'results': [
                {'date': '2025-07-01', 'datatype': 'PRCP', 'value': 0},
                {'date': '2025-07-02', 'datatype': 'PRCP', 'value': 10}
            ]
        }
        result = data_processor.process_noaa_data(raw_data, "NYC123")
        assert result is None

    def test_process_noaa_data_single_temperature(self, data_processor):
        """Test processing NOAA data with only TMAX or TMIN."""
        raw_data = {
            'results': [
                {'date': '2025-07-01', 'datatype': 'TMAX', 'value': 750}
            ]
        }
        result = data_processor.process_noaa_data(raw_data, "NYC123")
        assert result is not None
        assert len(result) == 1
        assert result['temperature_max'].iloc[0] == 750
        assert result['temperature_avg'].iloc[0] == 750
        assert 'temperature_min' not in result.columns or result['temperature_min'].isna().all()

    def test_process_eia_data_valid(self, data_processor, sample_eia_data):
        """Test processing valid EIA energy data."""
        result = data_processor.process_eia_data(sample_eia_data, "NYISO")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['date', 'energy_demand', 'eia_region', 'data_source', 'processed_at']
        assert result['energy_demand'].iloc[0] == 1000
        assert result['eia_region'].iloc[0] == 'NYISO'
        assert result['data_source'].iloc[0] == 'EIA'
        assert (pd.to_datetime(result['date']) == pd.to_datetime(['2025-07-01', '2025-07-02'])).all()

    def test_process_eia_data_empty(self, data_processor):
        """Test processing empty EIA data."""
        result = data_processor.process_eia_data({}, "NYISO")
        assert result is None
        
        result = data_processor.process_eia_data({'response': {'data': []}}, "NYISO")
        assert result is None

    def test_process_eia_data_missing_columns(self, data_processor):
        """Test processing EIA data with missing required columns."""
        raw_data = {
            'response': {
                'data': [
                    {'period': '2025-07-01', 'other_col': 1000}
                ]
            }
        }
        result = data_processor.process_eia_data(raw_data, "NYISO")
        assert result is None

  
    def test_process_city_data_weather_only(self, data_processor, sample_noaa_data):
        """Test processing city data with only weather data."""
        city = City(name="New York", state="NY", noaa_station="NYC123", eia_region="NYISO", lat=40.7128, lon=-74.0060)
        result = data_processor.process_city_data(sample_noaa_data, None, city)
        
        assert result is not None
        assert len(result) == 2
        assert 'temperature_avg' in result.columns
        assert 'energy_demand' in result.columns
        assert result['energy_demand'].isna().all()
        assert result['eia_region'].iloc[0] == 'NYISO'
        assert result['city'].iloc[0] == 'New York'

    def test_process_city_data_energy_only(self, data_processor, sample_eia_data):
        """Test processing city data with only energy data."""
        city = City(name="New York", state="NY", noaa_station="NYC123", eia_region="NYISO", lat=40.7128, lon=-74.0060)
        result = data_processor.process_city_data(None, sample_eia_data, city)
        
        assert result is not None
        assert len(result) == 2
        assert 'energy_demand' in result.columns
        assert 'temperature_avg' not in result.columns
        assert result['energy_demand'].iloc[0] == 1000
        assert result['city'].iloc[0] == 'New York'

    def test_process_city_data_none(self, data_processor):
        """Test processing city data with no data."""
        city = City(name="New York", state="NY", noaa_station="NYC123", eia_region="NYISO", lat=40.7128, lon=-74.0060)
        result = data_processor.process_city_data(None, None, city)
        assert result is None

    def test_check_data_quality_valid(self, data_processor):
        """Test data quality checks with valid data."""
        df = pd.DataFrame({
            'date': pd.date_range(start=datetime.now().date() - timedelta(days=2), periods=5, freq='D'),
            'city': ['New York'] * 5,
            'temperature_avg': [65.0, 66.0, 67.0, 68.0, 69.0],
            'energy_demand': [1000, 1010, 1020, 1030, 1040],
            'processed_at': [datetime.now()] * 5
        })
        result = data_processor.check_data_quality(df)
        
        assert result['passed'] is True
        assert len(result['issues']) == 0
        assert result['summary']['total_rows'] == 5
        assert result['summary']['missing_values'] == 0
        assert result['summary']['outliers'] == 0
        assert result['summary']['data_freshness_hours'] <= 48

    def test_check_data_quality_outliers(self, data_processor):
        """Test data quality checks with outliers."""
        df = pd.DataFrame({
            'date': pd.date_range(start=datetime.now().date() - timedelta(days=2), periods=5, freq='D'),
            'city': ['New York'] * 5,
            'temperature_avg': [65.0, 150.0, 67.0, -60.0, 69.0],  # Outliers: 150°F, -60°F
            'energy_demand': [1000, 1010, -10, 1030, 1040],  # Outlier: -10
            'processed_at': [datetime.now()] * 5
        })
        result = data_processor.check_data_quality(df)
        
        assert result['passed'] is False
        assert len(result['issues']) > 0
        assert result['summary']['outliers'] == 3  # 2 temperature outliers + 1 energy outlier
        assert any("High temperature outliers" in issue for issue in result['issues'])
        assert any("Low temperature outliers" in issue for issue in result['issues'])
        assert any("Negative energy values" in issue for issue in result['issues'])

    def test_check_data_quality_missing_values(self, data_processor):
        """Test data quality checks with missing values."""
        df = pd.DataFrame({
            'date': pd.date_range(start=datetime.now().date() - timedelta(days=2), periods=5, freq='D'),
            'city': ['New York'] * 5,
            'temperature_avg': [65.0, np.nan, 67.0, np.nan, 69.0],
            'energy_demand': [1000, 1010, np.nan, 1030, 1040],
            'processed_at': [datetime.now()] * 5
        })
        result = data_processor.check_data_quality(df)
        
        assert result['passed'] is False
        assert len(result['issues']) > 0
        assert result['summary']['missing_values'] == 3  # 2 temperature + 1 energy
        assert any("Missing values in temperature_avg" in issue for issue in result['issues'])
        assert any("Missing values in energy_demand" in issue for issue in result['issues'])

    def test_check_data_quality_stale_data(self, data_processor):
        """Test data quality checks with stale data."""
        df = pd.DataFrame({
            'date': pd.date_range(start='2025-06-01', periods=5, freq='D'),
            'city': ['New York'] * 5,
            'temperature_avg': [65.0, 66.0, 67.0, 68.0, 69.0],
            'energy_demand': [1000, 1010, 1020, 1030, 1040]
        })
        result = data_processor.check_data_quality(df)
        
        assert result['passed'] is False
        assert any("Data is stale" in issue for issue in result['issues'])
        assert result['summary']['data_freshness_hours'] > 48

    def test_check_data_quality_empty(self, data_processor):
        """Test data quality checks with empty DataFrame."""
        result = data_processor.check_data_quality(pd.DataFrame())
        
        assert result['passed'] is False
        assert "No data to check" in result['issues']
        assert result['summary']['total_rows'] == 0

    def test_generate_quality_summary(self, data_processor):
        """Test generating quality summary from multiple quality checks."""
        quality_checks = [
            {
                'passed': True,
                'issues': [],
                'summary': {'total_rows': 10, 'missing_values': 0, 'outliers': 0}
            },
            {
                'passed': False,
                'issues': ["Missing values in temperature_avg: 2 (20.0%)"],
                'summary': {'total_rows': 10, 'missing_values': 2, 'outliers': 1}
            }
        ]
        result = data_processor.generate_quality_summary(quality_checks)
        
        assert result['total_datasets_checked'] == 2
        assert result['datasets_passed'] == 1
        assert result['pass_rate_percent'] == 50.0
        assert result['total_data_rows'] == 20
        assert result['total_missing_values'] == 2
        assert result['total_outliers'] == 1
        assert result['unique_issue_types'] == 1
        assert 'generated_at' in result

    def test_generate_quality_summary_empty(self, data_processor):
        """Test generating quality summary with no checks."""
        result = data_processor.generate_quality_summary([])
        
        assert result['total_datasets_checked'] == 0
        assert result['datasets_passed'] == 0
        assert result['pass_rate_percent'] == 0
        assert result['total_data_rows'] == 0
        assert result['total_missing_values'] == 0
        assert result['total_outliers'] == 0
        assert result['unique_issue_types'] == 0

    def test_validate_data_consistency_weather(self, data_processor, sample_noaa_data):
        """Test validating consistency of processed weather data."""
        processed_df = pd.DataFrame({
            'date': pd.to_datetime(['2025-07-01', '2025-07-02']),
            'temperature_max': [750, 780],
            'temperature_min': [550, 580],
            'temperature_avg': [650, 680],
            'station_id': ['NYC123', 'NYC123'],
            'data_source': ['NOAA', 'NOAA'],
            'processed_at': [datetime.now()] * 2
        })
        result = data_processor.validate_data_consistency(sample_noaa_data, processed_df, 'weather')
        assert result is True

    def test_validate_data_consistency_weather_mismatch(self, data_processor, sample_noaa_data):
        """Test validating weather data with a mismatch."""
        processed_df = pd.DataFrame({
            'date': pd.to_datetime(['2025-07-01', '2025-07-02']),
            'temperature_max': [800, 780],  # Mismatch on first value
            'temperature_min': [550, 580],
            'temperature_avg': [675, 680],
            'station_id': ['NYC123', 'NYC123'],
            'data_source': ['NOAA', 'NOAA'],
            'processed_at': [datetime.now()] * 2
        })
        result = data_processor.validate_data_consistency(sample_noaa_data, processed_df, 'weather')
        assert result is False

    def test_validate_data_consistency_energy(self, data_processor, sample_eia_data):
        """Test validating consistency of processed energy data."""
        processed_df = pd.DataFrame({
            'date': pd.to_datetime(['2025-07-01', '2025-07-02']),
            'energy_demand': [1000, 1050],
            'eia_region': ['NYISO', 'NYISO'],
            'data_source': ['EIA', 'EIA'],
            'processed_at': [datetime.now()] * 2
        })
        result = data_processor.validate_data_consistency(sample_eia_data, processed_df, 'energy')
        assert result is True

    def test_validate_data_consistency_energy_mismatch(self, data_processor, sample_eia_data):
        """Test validating energy data with a mismatch."""
        processed_df = pd.DataFrame({
            'date': pd.to_datetime(['2025-07-01', '2025-07-02']),
            'energy_demand': [1100, 1050],  # Mismatch on first value
            'eia_region': ['NYISO', 'NYISO'],
            'data_source': ['EIA', 'EIA'],
            'processed_at': [datetime.now()] * 2
        })
        result = data_processor.validate_data_consistency(sample_eia_data, processed_df, 'energy')
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])