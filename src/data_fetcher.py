import requests
from loguru import logger
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dateutil.parser import parse
import pandas as pd

class Config:
    def __init__(self):
        self.data_paths = {
            'raw': 'data/raw',
            'processed': 'data/processed'
        }
        self.api_keys = {
            'noaa': os.getenv('NOAA_API_KEY', 'YOUR_NOAA_TOKEN'),
            'eia': os.getenv('EIA_API_KEY', 'YOUR_EIA_KEY')
        }
        self.rate_limits = {
            'retry_attempts': 3,
            'backoff_factor': 2.0,
            'noaa_requests_per_second': 5,
            'eia_requests_per_second': 5,
            'chunk_size_days': 30,
            'max_fetch_days': 90,
            'buffer_days': 2
        }
        self.cities = [
            City(name='New York', lat=40.7128, lon=-74.0060, noaa_station='GHCND:USW00094728', eia_region='NYIS'),
            City(name='Chicago', lat=41.8781, lon=-87.6298, noaa_station='GHCND:USW00094846', eia_region='PJM'),
            City(name='Houston', lat=29.7604, lon=-95.3698, noaa_station='GHCND:USW00012960', eia_region='ERCO'),
            City(name='Phoenix', lat=33.4484, lon=-112.0740, noaa_station='GHCND:USW00023183', eia_region='AZPS'),
            City(name='Seattle', lat=47.6062, lon=-122.3321, noaa_station='GHCND:USW00024233', eia_region='SCL')
        ]
    
    @staticmethod
    def load():
        return Config()

class City:
    def __init__(self, name: str, lat: float, lon: float, noaa_station: str, eia_region: str):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.noaa_station = noaa_station
        self.eia_region = eia_region

class DataFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Energy-Analysis-Pipeline/1.0'
        })
    
    def _validate_date(self, date_str: str) -> bool:
        try:
            parse(date_str)
            return True
        except ValueError:
            logger.error(f"Invalid date format: {date_str}")
            return False
    
    def _constrain_date_range(self, start_date: str, end_date: str) -> Tuple[str, str]:
        """Ensure date range respects max days and buffer period."""
        start_dt = parse(start_date).date()
        end_dt = parse(end_date).date()
        max_end = datetime.now().date() - timedelta(days=self.config.rate_limits['buffer_days'])
        
        if end_dt > max_end:
            logger.info(f"End date {end_date} adjusted to {max_end} (buffer: {self.config.rate_limits['buffer_days']} days)")
            end_dt = max_end
        
        max_start = end_dt - timedelta(days=self.config.rate_limits['max_fetch_days'] - 1)
        if start_dt < max_start:
            logger.info(f"Start date {start_date} adjusted to {max_start} (max: {self.config.rate_limits['max_fetch_days']} days)")
            start_dt = max_start
        
        return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
    
    def _make_request_with_retry(self, url: str, params: dict, headers: dict = None) -> Optional[requests.Response]:
        max_retries = self.config.rate_limits['retry_attempts']
        delay = 2.0
        for attempt in range(max_retries):
            try:
                logger.debug(f"Request attempt {attempt + 1}/{max_retries} to {url}")
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning(f"Rate limited on attempt {attempt + 1}, waiting...")
                    time.sleep(10)
                elif attempt < max_retries - 1:
                    logger.warning(f"HTTP error on attempt {attempt + 1}: {str(e)}")
                    time.sleep(delay)
                    delay *= self.config.rate_limits['backoff_factor']
                else:
                    logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request error on attempt {attempt + 1}: {str(e)}")
                    time.sleep(delay)
                    delay *= self.config.rate_limits['backoff_factor']
                else:
                    logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                    raise
        return None
    
    def fetch_noaa_data(self, station_id: str, start_date: str, end_date: str) -> Optional[Dict]:
        start_date, end_date = self._constrain_date_range(start_date, end_date)
        if not all(self._validate_date(d) for d in [start_date, end_date]):
            raise ValueError("Invalid date format, expected YYYY-MM-DD")
        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        params = {
            'datasetid': 'GHCND',
            'stationid': station_id,
            'startdate': start_date,
            'enddate': end_date,
            'datatypeid': 'TMAX,TMIN',
            'limit': 1000,
            'units': 'standard'
        }
        headers = {'token': self.config.api_keys['noaa']}
        logger.info(f"Fetching NOAA data for station {station_id} from {start_date} to {end_date}")
        try:
            response = self._make_request_with_retry(url, params, headers)
            if not response:
                return None
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"Invalid NOAA response format: expected dict, got {type(data)}")
            if 'results' in data:
                sample_temps = [
                    {'date': r['date'], 'datatype': r['datatype'], 'value': r['value'], 'station': r['station']}
                    for r in data['results'][:5]
                ]
                logger.debug(f"Sample NOAA data for {station_id}: {sample_temps}")
            self._save_raw_weather_data(data, station_id, start_date)
            time.sleep(1.0 / self.config.rate_limits['noaa_requests_per_second'])
            logger.info(f"Successfully fetched NOAA data for {station_id}")
            return data
        except Exception as e:
            logger.error(f"Failed fetching NOAA data for {station_id}: {str(e)}")
            return None
    
    def fetch_eia_data(self, region: str, start_date: str, end_date: str) -> Optional[Dict]:
        start_date, end_date = self._constrain_date_range(start_date, end_date)
        if not all(self._validate_date(d) for d in [start_date, end_date]):
            raise ValueError("Invalid date format, expected YYYY-MM-DD")
        url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
        params = {
            'api_key': self.config.api_keys['eia'],
            'frequency': 'daily',
            'data[0]': 'value',
            'facets[respondent][]': region,
            'start': start_date,
            'end': end_date,
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc'
        }
        logger.info(f"Fetching EIA data for {region} from {start_date} to {end_date}")
        try:
            response = self._make_request_with_retry(url, params)
            if not response:
                return None
            data = response.json()
            if not isinstance(data, dict) or 'response' not in data:
                logger.warning(f"Invalid EIA response for {region}, trying alternative...")
                return self._fetch_eia_alternative(region, start_date, end_date)
            logger.debug(f"Raw EIA data structure for {region}: {json.dumps(data.get('response', {}).get('data', [])[:2], indent=2)}")
            self._save_raw_energy_data(data, region, start_date)
            time.sleep(1.0 / self.config.rate_limits['eia_requests_per_second'])
            logger.info(f"Successfully fetched EIA data for {region}")
            return data
        except Exception as e:
            logger.error(f"Failed fetching EIA data for {region}: {str(e)}")
            return self._fetch_eia_alternative(region, start_date, end_date)
    
    def _fetch_eia_alternative(self, region: str, start_date: str, end_date: str) -> Optional[Dict]:
        start_date, end_date = self._constrain_date_range(start_date, end_date)
        url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
        params = {
            'api_key': self.config.api_keys['eia'],
            'frequency': 'daily',
            'data[0]': 'value',
            'facets[respondent][]': region,
            'start': start_date,
            'end': end_date
        }
        logger.info(f"Trying alternative EIA request for {region}")
        try:
            response = self._make_request_with_retry(url, params)
            if not response:
                return None
            data = response.json()
            if not isinstance(data, dict) or 'response' not in data:
                raise ValueError("Invalid EIA response structure")
            logger.debug(f"Raw alternative EIA data structure for {region}: {json.dumps(data.get('response', {}).get('data', [])[:2], indent=2)}")
            self._save_raw_energy_data(data, region, start_date)
            time.sleep(1.0 / self.config.rate_limits['eia_requests_per_second'])
            logger.info(f"Successfully fetched alternative EIA data for {region}")
            return data
        except Exception as e:
            logger.error(f"Alternative EIA request failed for {region}: {str(e)}")
            return None
    
    def _save_raw_weather_data(self, data: Dict, station_id: str, date: str) -> None:
        try:
            weather_dir = os.path.join(self.config.data_paths['raw'], 'weather')
            os.makedirs(weather_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"weather_{station_id.replace(':', '_')}_{date}_{timestamp}.json"
            filepath = os.path.join(weather_dir, filename)
            enhanced_data = {
                'metadata': {
                    'fetch_timestamp': datetime.now().isoformat(),
                    'data_type': 'weather',
                    'station_id': station_id,
                    'date_requested': date,
                    'api_source': 'NOAA'
                },
                'raw_data': data
            }
            with open(filepath, 'w') as f:
                json.dump(enhanced_data, f, indent=2, default=str)
            logger.debug(f"Saved weather raw data to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save weather raw data for {station_id}: {str(e)}")
    
    def _save_raw_energy_data(self, data: Dict, region: str, date: str) -> None:
        try:
            energy_dir = os.path.join(self.config.data_paths['raw'], 'energy')
            os.makedirs(energy_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"energy_{region}_{date}_{timestamp}.json"
            filepath = os.path.join(energy_dir, filename)
            enhanced_data = {
                'metadata': {
                    'fetch_timestamp': datetime.now().isoformat(),
                    'data_type': 'energy',
                    'region': region,
                    'date_requested': date,
                    'api_source': 'EIA'
                },
                'raw_data': data
            }
            with open(filepath, 'w') as f:
                json.dump(enhanced_data, f, indent=2, default=str)
            logger.debug(f"Saved energy raw data to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save energy raw data for {region}: {str(e)}")
    
    def fetch_city_data(self, city: City, start_date: str, end_date: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        weather_data = None
        energy_data = None
        try:
            weather_data = self.fetch_noaa_data(city.noaa_station, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch weather data for {city.name}: {str(e)}")
        try:
            energy_data = self.fetch_eia_data(city.eia_region, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch energy data for {city.name}: {str(e)}")
        return weather_data, energy_data
    
    
def fetch_historical_data(self, days: int = 90) -> List[Tuple[str, Optional[Dict], Optional[Dict]]]:
    days = min(days, self.config.rate_limits['max_fetch_days'])
    # Stop 2 days before current day to prevent availability issues
    end_date = datetime.now().date() - timedelta(days=2)
    start_date = end_date - timedelta(days=days - 1)
    logger.info(f"Fetching {days} days of historical data from {start_date} to {end_date}")
    results = []
    chunk_size = self.config.rate_limits['chunk_size_days']
    current_start = start_date
    while current_start <= end_date:
        chunk_end = min(current_start + timedelta(days=chunk_size - 1), end_date)
        logger.info(f"Processing chunk: {current_start} to {chunk_end}")
        for city in self.config.cities:
            try:
                weather_data, energy_data = self.fetch_city_data(
                    city,
                    current_start.strftime('%Y-%m-%d'),
                    chunk_end.strftime('%Y-%m-%d')
                )
                results.append((city.name, weather_data, energy_data))
                logger.debug(f"Fetched data for {city.name}: {current_start} to {chunk_end}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {city.name} in chunk {current_start}-{chunk_end}: {str(e)}")
                results.append((city.name, None, None))
        current_start = chunk_end + timedelta(days=1)
        if current_start <= end_date:
            logger.info("Waiting 5 seconds before next chunk...")
            time.sleep(5)
    logger.info(f"Historical data fetch complete: {len(results)} city records")
    return results