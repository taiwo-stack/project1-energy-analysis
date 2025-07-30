"""Configuration management for energy analysis pipeline."""
import os

import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

@dataclass
class City:
    """Represents a city with weather, energy data sources, and coordinates."""
    name: str
    state: str
    noaa_station: str
    eia_region: str
    lat: float
    lon: float

@dataclass
class Config:
    """Main configuration class for the energy analysis pipeline."""
    api_keys: Dict[str, str]
    data_paths: Dict[str, str]
    cities: List[City]
    quality_checks: Dict[str, Any]
    rate_limits: Dict[str, Any]
    logging: Dict[str, Any]
    city_colors: Dict[str, str]
    
    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> "Config":
        """
        Load configuration from YAML file and environment variables.
        
        Args:
            config_path: Path to config.yaml
        
        Returns:
            Config object
        """
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        api_keys = {
            'noaa': os.getenv('NOAA_API_KEY', data.get('api_keys', {}).get('noaa', '')),
            'eia': os.getenv('EIA_API_KEY', data.get('api_keys', {}).get('eia', ''))
        }
        
        cities = [City(**city_data) for city_data in data.get('cities', [])]
        
        city_colors = data.get('city_colors', {
            'New York': '#1f77b4',
            'Chicago': '#ff7f0e',
            'Houston': '#2ca02c',
            'Phoenix': '#d62728',
            'Seattle': '#9467bd'
        })
        
        # Set default rate limits with all required values
        rate_limits = data.get('rate_limits', {})
        rate_limits.setdefault('max_fetch_days', 90)
        rate_limits.setdefault('buffer_days', 3)
        rate_limits.setdefault('processing_delay_seconds', 2)
        rate_limits.setdefault('noaa_requests_per_second', 5)
        rate_limits.setdefault('eia_requests_per_second', 1)
        rate_limits.setdefault('retry_attempts', 5)
        rate_limits.setdefault('backoff_factor', 1.0)
        rate_limits.setdefault('chunk_size_days', 30)
        
        # Set default logging configuration
        logging_config = data.get('logging', {})
        logging_config.setdefault('level', 'INFO')
        logging_config.setdefault('rotation', '10 MB')
        logging_config.setdefault('retention', '7 days')
        
        # Set default quality checks
        quality_checks = data.get('quality_checks', {})
        quality_checks.setdefault('temperature', {
            'max_fahrenheit': 130,
            'min_fahrenheit': -50
        })
        quality_checks.setdefault('energy', {'min_value': 0})
        quality_checks.setdefault('freshness', {'max_age_hours': 48})
        quality_checks.setdefault('completeness', {'max_missing_percentage': 10})
        
        return cls(
            api_keys=api_keys,
            data_paths=data.get('data_paths', {
                'raw': 'data/raw',
                'processed': 'data/processed',
                'logs': 'logs',
                'cache': 'data/cache'
            }),
            cities=cities,
            quality_checks=quality_checks,
            rate_limits=rate_limits,
            logging=logging_config,
            city_colors=city_colors
        )
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate API keys
        if not self.api_keys.get('noaa'):
            errors.append("NOAA API key is missing (set NOAA_API_KEY in .env or config.yaml)")
        if not self.api_keys.get('eia'):
            errors.append("EIA API key is missing (set EIA_API_KEY in .env or config.yaml)")
        
        # Validate data paths
        required_paths = ['raw', 'processed', 'logs', 'cache']
        for path_name in required_paths:
            if path_name not in self.data_paths:
                errors.append(f"Missing required data path: {path_name}")
                continue
                
            path = self.data_paths[path_name]
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created/verified directory: {path}")
            except Exception as e:
                errors.append(f"Cannot create {path_name} directory {path}: {str(e)}")
        
        # Validate cities
        if not self.cities:
            errors.append("No cities configured in config.yaml")
        
        for city in self.cities:
            if not all([city.name, city.noaa_station, city.eia_region]):
                errors.append(f"Incomplete configuration for city {city.name}: missing name, noaa_station, or eia_region")
            if not isinstance(city.lat, (int, float)) or not isinstance(city.lon, (int, float)):
                errors.append(f"Invalid coordinates for city {city.name}: lat={city.lat}, lon={city.lon}")
        
        # Validate rate limits
        required_rate_limit_keys = [
            'noaa_requests_per_second', 'eia_requests_per_second', 'retry_attempts', 
            'backoff_factor', 'chunk_size_days', 'max_fetch_days', 'buffer_days',
            'processing_delay_seconds'
        ]
        
        for key in required_rate_limit_keys:
            if key not in self.rate_limits:
                errors.append(f"Missing rate limit configuration: {key}")
            elif not isinstance(self.rate_limits[key], (int, float)) or self.rate_limits[key] <= 0:
                errors.append(f"Invalid rate limit value for {key}: {self.rate_limits[key]} (must be positive number)")
        
        # Validate specific rate limit constraints
        if 'max_fetch_days' in self.rate_limits and 'buffer_days' in self.rate_limits:
            if self.rate_limits['buffer_days'] >= self.rate_limits['max_fetch_days']:
                errors.append(f"buffer_days ({self.rate_limits['buffer_days']}) must be less than max_fetch_days ({self.rate_limits['max_fetch_days']})")
        
        # Validate quality checks
        required_quality_checks = ['temperature', 'energy', 'freshness', 'completeness']
        for key in required_quality_checks:
            if key not in self.quality_checks:
                errors.append(f"Missing quality check configuration: {key}")
        
        # Validate temperature thresholds
        if 'temperature' in self.quality_checks:
            temp_config = self.quality_checks['temperature']
            if 'max_fahrenheit' not in temp_config or 'min_fahrenheit' not in temp_config:
                errors.append("Temperature quality check missing max_fahrenheit or min_fahrenheit")
            elif temp_config['min_fahrenheit'] >= temp_config['max_fahrenheit']:
                errors.append("Temperature min_fahrenheit must be less than max_fahrenheit")
        
        # Validate completeness percentage
        if 'completeness' in self.quality_checks:
            completeness_config = self.quality_checks['completeness']
            if 'max_missing_percentage' in completeness_config:
                max_missing = completeness_config['max_missing_percentage']
                if not isinstance(max_missing, (int, float)) or max_missing < 0 or max_missing > 100:
                    errors.append("max_missing_percentage must be a number between 0 and 100")
        
        # Validate logging configuration
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if not self.logging.get('level') in valid_log_levels:
            errors.append(f"Invalid logging level: {self.logging.get('level', 'N/A')} (must be one of {valid_log_levels})")
        
        # Validate city colors
        if not self.city_colors:
            errors.append("No city colors defined in config.yaml")
        for city in self.cities:
            if city.name not in self.city_colors:
                errors.append(f"No color defined for city {city.name}")
        
        if errors:
            logger.error(f"Configuration validation failed with {len(errors)} errors")
        else:
            logger.info("Configuration validation passed successfully")
        
        return errors
    
    def get_city_by_name(self, name: str) -> Optional[City]:
        """
        Get city configuration by name.
        
        Args:
            name: City name
        
        Returns:
            City object or None if not found
        """
        for city in self.cities:
            if city.name.lower() == name.lower():
                return city
        logger.warning(f"City not found: {name}")
        return None
    
    def get_max_fetch_days(self) -> int:
        """Get the maximum number of days that can be fetched."""
        return self.rate_limits['max_fetch_days']
    
    def get_buffer_days(self) -> int:
        """Get the number of buffer days."""
        return self.rate_limits['buffer_days']
    
    def get_processing_delay(self) -> float:
        """Get the processing delay in seconds."""
        return self.rate_limits['processing_delay_seconds']