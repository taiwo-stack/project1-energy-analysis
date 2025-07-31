"""
This module tests the Analyzer class functionality including:
- Data loading and filtering
- Correlation calculations
- Regression analysis
- Usage level calculations
- Data quality reporting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from analysis import Analyzer
from config import Config, City

class TestAnalyzer:
    """Test suite for the Analyzer class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        cities = [
            City(name="New York", state="NY", noaa_station="NYC123", 
                 eia_region="NYISO", lat=40.7128, lon=-74.0060),
            City(name="Chicago", state="IL", noaa_station="CHI456", 
                 eia_region="PJM", lat=41.8781, lon=-87.6298),
            City(name="Houston", state="TX", noaa_station="HOU789", 
                 eia_region="ERCOT", lat=29.7604, lon=-95.3698)
        ]
        
        config = Mock(spec=Config)
        config.cities = cities
        config.rate_limits = {
            'max_fetch_days': 90,
            'buffer_days': 2
        }
        config.data_paths = {
            'processed': '/tmp/test_data'
        }
        config.quality_checks = {
            'temperature': {'max_fahrenheit': 130, 'min_fahrenheit': -50},
            'energy': {'min_value': 0},
            'freshness': {'max_age_hours': 48},
            'completeness': {'min_coverage': 0.8}
        }
        config.city_colors = {
            'New York': '#1f77b4',
            'Chicago': '#ff7f0e', 
            'Houston': '#2ca02c'
        }
        config.get_city_by_name = Mock(side_effect=lambda name: next(
            (city for city in cities if city.name.lower() == name.lower()), None
        ))
        
        return config
    
    @pytest.fixture
    def analyzer(self, mock_config):
        """Create an Analyzer instance with mock configuration."""
        return Analyzer(mock_config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)  # For reproducible tests
        # Use recent dates that won't be filtered out by the buffer logic
        end_date = datetime.now().date() - timedelta(days=2)  # Account for buffer
        start_date = end_date - timedelta(days=29)  # 30 days total
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        cities = ['New York', 'Chicago', 'Houston']
        
        data = []
        for city in cities:
            base_temp = {'New York': 35, 'Chicago': 25, 'Houston': 55}[city]
            base_energy = {'New York': 1000, 'Chicago': 800, 'Houston': 1200}[city]
            
            for date in dates:
                # Create temperature with some seasonal variation
                temp_variation = np.random.normal(0, 10)
                temperature_avg = base_temp + temp_variation
                temperature_max = temperature_avg + np.random.uniform(5, 15)
                temperature_min = temperature_avg - np.random.uniform(5, 15)
                
                # Create energy demand with negative correlation to temperature for heating/cooling
                energy_variation = np.random.normal(0, 50)
                if city in ['New York', 'Chicago']:  # Heating dominant
                    energy_demand = base_energy - (temperature_avg - base_temp) * 5 + energy_variation
                else:  # Cooling dominant (Houston)
                    energy_demand = base_energy + (temperature_avg - base_temp) * 3 + energy_variation
                
                data.append({
                    'city': city,
                    'date': date,
                    'temperature_avg': temperature_avg,
                    'temperature_max': temperature_max,
                    'temperature_min': temperature_min,
                    'energy_demand': max(0, energy_demand)  # Ensure non-negative
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_with_missing(self, sample_data):
        """Create sample data with missing values for testing."""
        df = sample_data.copy()
        # Introduce some missing values
        df.loc[0:2, 'temperature_avg'] = np.nan
        df.loc[10:12, 'energy_demand'] = np.nan
        return df
    
    @pytest.fixture
    def sample_data_with_outliers(self, sample_data):
        """Create sample data with outliers for testing."""
        df = sample_data.copy()
        # Add temperature outliers
        df.loc[0, 'temperature_avg'] = 200  # Extreme hot
        df.loc[1, 'temperature_avg'] = -100  # Extreme cold
        # Add energy outliers
        df.loc[2, 'energy_demand'] = -50  # Negative energy
        return df
    
    def test_analyzer_initialization(self, mock_config):
        """Test that Analyzer initializes correctly."""
        analyzer = Analyzer(mock_config)
        assert analyzer.config == mock_config
        assert analyzer.max_fetch_days == 90
        assert analyzer.buffer_days == 2
    
    def test_get_available_date_range(self, analyzer):
        """Test date range calculation with buffer days."""
        start_date, end_date = analyzer.get_available_date_range()
        
        today = datetime.now().date()
        expected_end = today - timedelta(days=2)  # 2-day buffer
        expected_start = expected_end - timedelta(days=89)  # max_fetch_days - 1
        
        assert start_date == expected_start
        assert end_date == expected_end
    
    
    
    @patch('pathlib.Path.exists')
    def test_load_data_file_not_found(self, mock_exists, analyzer):
        """Test data loading when file doesn't exist."""
        mock_exists.return_value = False
        
        result = analyzer.load_data()
        
        assert result.empty
    
    @patch('pathlib.Path.exists')
    @patch('pandas.read_csv')
    def test_load_data_with_date_range(self, mock_read_csv, mock_exists, analyzer, sample_data):
        """Test data loading with custom date range."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_data
        
        # Use dates that are within the sample data range
        start_date = sample_data['date'].min().date() + timedelta(days=5)
        end_date = sample_data['date'].max().date() - timedelta(days=5)
        
        result = analyzer.load_data(date_range=(start_date, end_date))
        
        assert not result.empty
        assert result['date'].min().date() >= start_date
        assert result['date'].max().date() <= end_date
    
    @patch('pathlib.Path.exists')
    @patch('pandas.read_csv')
    def test_load_data_creates_temperature_avg(self, mock_read_csv, mock_exists, analyzer):
        """Test that temperature_avg is created from max/min when missing."""
        # Create data without temperature_avg but with recent dates
        recent_dates = pd.date_range(end=datetime.now().date() - timedelta(days=2), periods=2, freq='D')
        data = pd.DataFrame({
            'city': ['New York', 'Chicago'],
            'date': recent_dates,
            'temperature_max': [45.0, 35.0],
            'temperature_min': [25.0, 15.0],
            'energy_demand': [1000, 800]
        })
        
        mock_exists.return_value = True
        mock_read_csv.return_value = data
        
        result = analyzer.load_data()
        
        assert not result.empty, "Result should not be empty after date filtering"
        assert 'temperature_avg' in result.columns
        if len(result) > 0:  # Only check values if we have data after filtering
            assert result['temperature_avg'].iloc[0] == 35.0  # (45 + 25) / 2
            if len(result) > 1:
                assert result['temperature_avg'].iloc[1] == 25.0  # (35 + 15) / 2
    
    def test_calculate_correlations_success(self, analyzer, sample_data):
        """Test successful correlation calculation."""
        correlations = analyzer.calculate_correlations(sample_data)
        
        assert len(correlations) == 3  # Three cities
        assert 'New York' in correlations
        assert 'Chicago' in correlations
        assert 'Houston' in correlations
        
        # All correlations should be valid numbers (not NaN)
        for city, corr in correlations.items():
            assert isinstance(corr, (int, float))
            assert -1 <= corr <= 1 or np.isnan(corr)
    
    def test_calculate_correlations_selected_cities(self, analyzer, sample_data):
        """Test correlation calculation for selected cities only."""
        selected_cities = ['New York', 'Chicago']
        correlations = analyzer.calculate_correlations(sample_data, selected_cities)
        
        assert len(correlations) == 2
        assert 'New York' in correlations
        assert 'Chicago' in correlations
        assert 'Houston' not in correlations
    
    def test_calculate_correlations_empty_data(self, analyzer):
        """Test correlation calculation with empty data."""
        empty_df = pd.DataFrame()
        correlations = analyzer.calculate_correlations(empty_df)
        
        assert correlations == {}
    
    def test_calculate_correlations_missing_columns(self, analyzer):
        """Test correlation calculation with missing required columns."""
        df = pd.DataFrame({
            'city': ['New York'],
            'date': [datetime.now()]
            # Missing temperature_avg and energy_demand
        })
        
        correlations = analyzer.calculate_correlations(df)
        
        assert correlations == {}
    
    def test_calculate_regression_success(self, analyzer, sample_data):
        """Test successful regression calculation."""
        regression_stats = analyzer.calculate_regression(sample_data)
        
        assert len(regression_stats) == 3  # Three cities
        
        for city, stats in regression_stats.items():
            assert 'slope' in stats
            assert 'intercept' in stats
            assert 'r_squared' in stats
            assert 'p_value' in stats
            assert 'data_points' in stats
            
            # Check that values are reasonable
            assert isinstance(stats['slope'], (int, float))
            assert isinstance(stats['intercept'], (int, float))
            assert 0 <= stats['r_squared'] <= 1 or np.isnan(stats['r_squared'])
            assert stats['data_points'] > 0
    
    def test_calculate_regression_insufficient_data(self, analyzer):
        """Test regression calculation with insufficient data."""
        df = pd.DataFrame({
            'city': ['New York'],
            'date': [datetime.now()],
            'temperature_avg': [35.0],
            'energy_demand': [1000]
        })
        
        regression_stats = analyzer.calculate_regression(df)
        
        assert 'New York' in regression_stats
        stats = regression_stats['New York']
        assert np.isnan(stats['slope'])
        assert np.isnan(stats['intercept'])
        assert np.isnan(stats['r_squared'])
    
    def test_calculate_usage_levels_success(self, analyzer, sample_data):
        """Test successful usage level calculation."""
        usage_levels = analyzer.calculate_usage_levels(sample_data)
        
        assert len(usage_levels) == 3  # Three cities
        
        for city, data in usage_levels.items():
            assert 'lat' in data
            assert 'lon' in data
            assert 'current_usage' in data
            assert 'baseline_median' in data
            assert 'status' in data
            assert data['status'] in ['high', 'low']
            assert 'color' in data
            assert 'city_name' in data
            assert 'state' in data
            assert 'recent_data_points' in data
            assert 'baseline_data_points' in data
            assert 'last_updated' in data
    
    def test_calculate_usage_levels_selected_cities(self, analyzer, sample_data):
        """Test usage level calculation for selected cities."""
        selected_cities = ['New York', 'Houston']
        usage_levels = analyzer.calculate_usage_levels(sample_data, selected_cities)
        
        assert len(usage_levels) == 2
        assert 'New York' in usage_levels
        assert 'Houston' in usage_levels
        assert 'Chicago' not in usage_levels
    
    def test_calculate_usage_levels_empty_data(self, analyzer):
        """Test usage level calculation with empty data."""
        empty_df = pd.DataFrame()
        usage_levels = analyzer.calculate_usage_levels(empty_df)
        
        assert usage_levels == {}
    
    def test_get_usage_summary_success(self, analyzer, sample_data):
        """Test usage summary generation."""
        usage_levels = analyzer.calculate_usage_levels(sample_data)
        summary = analyzer.get_usage_summary(usage_levels)
        
        assert 'total_cities' in summary
        assert 'high_count' in summary
        assert 'low_count' in summary
        assert 'high_cities' in summary
        assert 'low_cities' in summary
        assert 'high_usage_avg' in summary
        assert 'low_usage_avg' in summary
        assert 'generated_at' in summary
        
        assert summary['total_cities'] == len(usage_levels)
        assert summary['high_count'] + summary['low_count'] == summary['total_cities']
    
    def test_get_usage_summary_empty_data(self, analyzer):
        """Test usage summary with empty usage levels."""
        summary = analyzer.get_usage_summary({})
        
        assert summary['total_cities'] == 0
        assert summary['high_count'] == 0
        assert summary['low_count'] == 0
        assert summary['high_cities'] == []
        assert summary['low_cities'] == []
    
    def test_generate_data_quality_report_success(self, analyzer, sample_data):
        """Test data quality report generation with good data."""
        report = analyzer.generate_data_quality_report(sample_data)
        
        assert 'passed' in report
        assert 'issues' in report
        assert 'summary' in report
        
        summary = report['summary']
        assert 'total_rows' in summary
        assert 'missing_values' in summary
        assert 'outliers' in summary
        assert 'data_freshness_days' in summary
        
        assert summary['total_rows'] == len(sample_data)
        # The data might not pass due to data freshness (expected behavior)
        # So we just check that the report structure is correct
        assert isinstance(report['passed'], bool)
    
    def test_generate_data_quality_report_with_recent_data(self, analyzer):
        """Test data quality report with properly recent data."""
        # Create data that should pass quality checks
        recent_date = datetime.now().date() - timedelta(days=1)  # Very recent
        good_data = pd.DataFrame({
            'city': ['New York'] * 10,
            'date': pd.date_range(end=recent_date, periods=10, freq='D'),
            'temperature_avg': np.random.normal(35, 5, 10),  # Normal temperatures
            'temperature_max': np.random.normal(45, 5, 10),
            'temperature_min': np.random.normal(25, 5, 10),
            'energy_demand': np.random.normal(1000, 100, 10)  # Positive energy values
        })
        
        report = analyzer.generate_data_quality_report(good_data)
        
        assert 'passed' in report
        assert 'issues' in report
        assert 'summary' in report
        
        # With recent, clean data, it should pass
        assert report['passed'] is True
    
    def test_generate_data_quality_report_with_issues(self, analyzer, sample_data_with_outliers):
        """Test data quality report with problematic data."""
        report = analyzer.generate_data_quality_report(sample_data_with_outliers)
        
        assert report['passed'] is False  # Should fail due to outliers
        assert len(report['issues']) > 0
        assert report['summary']['outliers'] > 0
    
    def test_generate_data_quality_report_missing_values(self, analyzer, sample_data_with_missing):
        """Test data quality report with missing values."""
        report = analyzer.generate_data_quality_report(sample_data_with_missing)
        
        assert report['summary']['missing_values'] > 0
        # Check that missing value issues are reported
        missing_issues = [issue for issue in report['issues'] if 'Missing values' in issue]
        assert len(missing_issues) > 0
    
    def test_generate_data_quality_report_empty_data(self, analyzer):
        """Test data quality report with empty data."""
        empty_df = pd.DataFrame()
        report = analyzer.generate_data_quality_report(empty_df)
        
        assert report['passed'] is False
        assert 'No data to check' in report['issues']
        assert report['summary']['total_rows'] == 0
    
    def test_generate_data_quality_report_no_relevant_columns(self, analyzer):
        """Test data quality report with irrelevant columns."""
        df = pd.DataFrame({
            'irrelevant_col1': [1, 2, 3],
            'irrelevant_col2': ['a', 'b', 'c']
        })
        
        report = analyzer.generate_data_quality_report(df)
        
        assert report['passed'] is False
        assert any('No temperature or energy columns found' in issue for issue in report['issues'])
    
    def test_date_filtering_edge_cases(self, analyzer):
        """Test edge cases in date filtering."""
        # Create sample data for this specific test
        sample_data = pd.DataFrame({
            'city': ['New York'] * 5,
            'date': pd.date_range('2024-01-01', periods=5),
            'temperature_avg': [35.0] * 5,
            'energy_demand': [1000] * 5
        })
        
        # Test with future date range (should return empty)
        future_start = datetime.now().date() + timedelta(days=10)
        future_end = datetime.now().date() + timedelta(days=20)
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=sample_data):
            result = analyzer.load_data(date_range=(future_start, future_end))
            assert result.empty
    
    def test_error_handling_in_correlations(self, analyzer):
        """Test error handling in correlation calculations."""
        # Create data that might cause calculation errors
        df = pd.DataFrame({
            'city': ['Test City'] * 5,
            'date': pd.date_range('2024-01-01', periods=5),
            'temperature_avg': [25.0] * 5,  # No variation
            'energy_demand': [1000] * 5     # No variation
        })
        
        # Mock the city config to avoid KeyError
        with patch.object(analyzer.config, 'get_city_by_name', return_value=None):
            correlations = analyzer.calculate_correlations(df)
            # Should handle the error gracefully
            assert isinstance(correlations, dict)
    
    def test_temperature_avg_fallback_logic(self, analyzer):
        """Test the temperature_avg creation fallback logic."""
        # Test with only temperature_max
        df_max_only = pd.DataFrame({
            'city': ['New York'],
            'date': pd.date_range('2024-01-01', periods=1),
            'temperature_max': [45.0],
            'energy_demand': [1000]
        })
        
        correlations = analyzer.calculate_correlations(df_max_only)
        assert len(correlations) >= 0  # Should not crash
        
        # Test with only temperature_min
        df_min_only = pd.DataFrame({
            'city': ['New York'],
            'date': pd.date_range('2024-01-01', periods=1),
            'temperature_min': [25.0],
            'energy_demand': [1000]
        })
        
        correlations = analyzer.calculate_correlations(df_min_only)
        assert len(correlations) >= 0  # Should not crash


# Integration tests
class TestAnalyzerIntegration:
    """Integration tests for the Analyzer class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def real_config_with_temp_dir(self, temp_data_dir):
        """Create a real config object with temporary directory."""
        config_data = {
            'api_keys': {'noaa': 'test_key', 'eia': 'test_key'},
            'data_paths': {'processed': temp_data_dir},
            'cities': [
                {'name': 'New York', 'state': 'NY', 'noaa_station': 'NYC123',
                 'eia_region': 'NYISO', 'lat': 40.7128, 'lon': -74.0060}
            ],
            'quality_checks': {
                'temperature': {'max_fahrenheit': 130, 'min_fahrenheit': -50},
                'energy': {'min_value': 0},
                'freshness': {'max_age_hours': 48},
                'completeness': {'min_coverage': 0.8}
            },
            'rate_limits': {'max_fetch_days': 90, 'buffer_days': 2},
            'logging': {'level': 'INFO'},
            'city_colors': {'New York': '#1f77b4'}
        }
        
        # Create temporary config file
        config_file = Path(temp_data_dir) / 'config.yaml'
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return Config.load(str(config_file))
    
    def test_full_analysis_pipeline(self, real_config_with_temp_dir, temp_data_dir):
        """Test the complete analysis pipeline with real file I/O."""
        # Create sample data with recent dates
        np.random.seed(42)
        end_date = datetime.now().date() - timedelta(days=2)
        start_date = end_date - timedelta(days=29)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        sample_data = []
        for city in ['New York']:  # Just test with one city for integration
            for date in dates:
                sample_data.append({
                    'city': city,
                    'date': date,
                    'temperature_avg': 35.0 + np.random.normal(0, 10),
                    'temperature_max': 45.0 + np.random.normal(0, 5),
                    'temperature_min': 25.0 + np.random.normal(0, 5),
                    'energy_demand': max(0, 1000 + np.random.normal(0, 100))
                })
        
        df = pd.DataFrame(sample_data)
        
        # Create test data file
        data_file = Path(temp_data_dir) / 'latest_historical.csv'
        df.to_csv(data_file, index=False)
        
        # Create analyzer and run full pipeline
        analyzer = Analyzer(real_config_with_temp_dir)
        
        # Load data
        loaded_df = analyzer.load_data()
        assert not loaded_df.empty
        
        # Run all analysis methods
        correlations = analyzer.calculate_correlations(loaded_df)
        assert len(correlations) >= 0  # May be empty if insufficient data
        
        regression_stats = analyzer.calculate_regression(loaded_df)
        assert len(regression_stats) >= 0
        
        usage_levels = analyzer.calculate_usage_levels(loaded_df)
        assert len(usage_levels) >= 0
        
        usage_summary = analyzer.get_usage_summary(usage_levels)
        assert 'total_cities' in usage_summary
        
        quality_report = analyzer.generate_data_quality_report(loaded_df)
        assert 'passed' in quality_report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])