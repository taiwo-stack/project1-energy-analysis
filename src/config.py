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
        
        # Set default rate limits
        rate_limits = data.get('rate_limits', {})
        rate_limits.setdefault('max_fetch_days', 90)
        rate_limits.setdefault('buffer_days', 2)
        
        return cls(
            api_keys=api_keys,
            data_paths=data.get('data_paths', {}),
            cities=cities,
            quality_checks=data.get('quality_checks', {}),
            rate_limits=rate_limits,
            logging=data.get('logging', {}),
            city_colors=city_colors
        )
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not self.api_keys.get('noaa'):
            errors.append("NOAA API key is missing (set NOAA_API_KEY in .env or config.yaml)")
        if not self.api_keys.get('eia'):
            errors.append("EIA API key is missing (set EIA_API_KEY in .env or config.yaml)")
            
        for path_name, path in self.data_paths.items():
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created/verified directory: {path}")
            except Exception as e:
                errors.append(f"Cannot create {path_name} directory {path}: {str(e)}")
        
        if not self.cities:
            errors.append("No cities configured in config.yaml")
        
        for city in self.cities:
            if not all([city.name, city.noaa_station, city.eia_region]):
                errors.append(f"Incomplete configuration for city {city.name}: missing name, noaa_station, or eia_region")
            if not isinstance(city.lat, (int, float)) or not isinstance(city.lon, (int, float)):
                errors.append(f"Invalid coordinates for city {city.name}: lat={city.lat}, lon={city.lon}")
        
        for key in ['noaa_requests_per_second', 'eia_requests_per_second', 'retry_attempts', 'backoff_factor', 'chunk_size_days', 'max_fetch_days', 'buffer_days']:
            if key not in self.rate_limits or not isinstance(self.rate_limits[key], (int, float)) or self.rate_limits[key] <= 0:
                errors.append(f"Invalid or missing rate limit: {key}")
        
        for key in ['temperature', 'energy', 'freshness', 'completeness']:
            if key not in self.quality_checks:
                errors.append(f"Missing quality check configuration: {key}")
        
        if not self.logging.get('level') in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            errors.append(f"Invalid logging level: {self.logging.get('level', 'N/A')}")
        
        if not self.city_colors:
            errors.append("No city colors defined in config.yaml")
        for city in self.cities:
            if city.name not in self.city_colors:
                errors.append(f"No color defined for city {city.name}")
        
        if errors:
            logger.error(f"Configuration validation failed with {len(errors)} errors")
        
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