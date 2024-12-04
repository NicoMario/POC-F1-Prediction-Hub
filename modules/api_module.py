# api_module.py
import requests
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import json
import time
from datetime import datetime, timedelta

class F1API:
    """
    Handles all F1 data retrieval and processing operations using the Ergast F1 API.
    Includes caching, rate limiting, and comprehensive error handling.
    """
    
    # Mapping between standard GP names and API names
    GP_NAME_MAPPING = {
        'Bahrain GP': 'Bahrain Grand Prix',
        'Saudi Arabian GP': 'Saudi Arabian Grand Prix',
        'Australian GP': 'Australian Grand Prix',
        'Azerbaijan GP': 'Azerbaijan Grand Prix',
        'Miami GP': 'Miami Grand Prix',
        'Monaco GP': 'Monaco Grand Prix',
        'Spanish GP': 'Spanish Grand Prix',
        'Canadian GP': 'Canadian Grand Prix',
        'Austrian GP': 'Austrian Grand Prix',
        'British GP': 'British Grand Prix',
        'Hungarian GP': 'Hungarian Grand Prix',
        'Belgian GP': 'Belgian Grand Prix',
        'Dutch GP': 'Dutch Grand Prix',
        'Italian GP': 'Italian Grand Prix',
        'Singapore GP': 'Singapore Grand Prix',
        'Japanese GP': 'Japanese Grand Prix',
        'Qatar GP': 'Qatar Grand Prix',
        'US GP': 'United States Grand Prix',
        'Mexican GP': 'Mexican Grand Prix',
        'Brazilian GP': 'Brazilian Grand Prix',
        'Las Vegas GP': 'Las Vegas Grand Prix',
        'Abu Dhabi GP': 'Abu Dhabi Grand Prix'
    }
    
    def __init__(self) -> None:
        """Initialize the F1 API with logging, caching, and rate limiting."""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('f1_api.log')
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # API configuration
        self.base_url = 'http://ergast.com/api/f1'
        
        # Cache setup
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Rate limiting
        self.rate_limit = 0.1  # seconds between requests
        self.last_request = 0
        
        self.logger.info("F1 API initialized successfully")

    def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def _get_cache_path(self, endpoint: str, params: Dict[str, Any]) -> Path:
        """Generate a cache file path for the given endpoint and parameters."""
        param_string = json.dumps(params, sort_keys=True)
        cache_key = f"{endpoint}_{hash(param_string)}.json"
        return self.cache_dir / cache_key

    def _get_cached_data(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Retrieve cached data if available and not expired."""
        if cache_path.exists():
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < 24 * 3600:  # 24 hour cache validity
                try:
                    with cache_path.open('r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning(f"Corrupted cache file: {cache_path}")
                    cache_path.unlink()
        return None

    def _cache_data(self, cache_path: Path, data: Dict[str, Any]) -> None:
        """Cache the retrieved data."""
        try:
            with cache_path.open('w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Error caching data: {str(e)}")

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an API request with rate limiting and error handling."""
        self._respect_rate_limit()
        
        url = f"{self.base_url}/{endpoint}.json"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise

    def _safe_int_convert(self, value: Any) -> int:
        """Safely convert a value to integer, returning 0 if conversion fails."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def get_race_results(self, 
                        year: int, 
                        grand_prix: Optional[List[str]] = None,
                        weather: Optional[List[str]] = None,
                        include_sprint: bool = True) -> pd.DataFrame:
        """Get race results with comprehensive filtering options."""
        self.logger.info(f"Fetching race results for year {year}")
        if grand_prix:
            self.logger.info(f"Filtering for GPs: {grand_prix}")
            api_grand_prix = [self.GP_NAME_MAPPING.get(gp, gp) for gp in grand_prix]
            self.logger.info(f"Corresponding API names: {api_grand_prix}")

        try:
            data = self._make_request(f"{year}/results")
            
            if 'MRData' not in data or 'RaceTable' not in data['MRData']:
                self.logger.error(f"Invalid API response format for year {year}")
                return pd.DataFrame()

            races = data['MRData']['RaceTable'].get('Races', [])
            
            self.logger.info(f"Found {len(races)} races for year {year}")
            
            results = []
            for race in races:
                race_name = race['raceName']
                
                # Filter by Grand Prix if specified
                if grand_prix:
                    if race_name not in api_grand_prix:
                        continue
                
                race_results = race.get('Results', [])
                self.logger.debug(f"Processing {len(race_results)} results for {race_name}")
                
                for result in race_results:
                    try:
                        # Basic required fields
                        result_data = {
                            'year': year,
                            'race': next((k for k, v in self.GP_NAME_MAPPING.items() if v == race_name), race_name),
                            'round': int(race['round']),
                            'circuit': race['Circuit']['circuitName'],
                            'grid': self._safe_int_convert(result.get('grid', '0')),
                            'position': self._safe_int_convert(result.get('position', '0')),
                            'points': float(result.get('points', 0)),
                            'laps': self._safe_int_convert(result.get('laps', '0')),
                            'status': result.get('status', '')
                        }

                        # Add driver info
                        if 'Driver' in result:
                            driver = result['Driver']
                            result_data['driver'] = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
                            result_data['driver_number'] = driver.get('permanentNumber', '')
                            result_data['driver_code'] = driver.get('code', '')

                        # Add constructor info
                        if 'Constructor' in result:
                            result_data['constructor'] = result['Constructor'].get('name', 'Unknown')

                        # Add timing info if available
                        if 'Time' in result:
                            result_data['race_time'] = result['Time'].get('time', '')

                        # Add fastest lap info if available
                        if 'FastestLap' in result:
                            result_data['fastest_lap_rank'] = self._safe_int_convert(
                                result['FastestLap'].get('rank', '0'))
                            if 'Time' in result['FastestLap']:
                                result_data['fastest_lap_time'] = result['FastestLap']['Time'].get('time', '')

                        results.append(result_data)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing result in race {race_name}: {str(e)}")
                        continue

            df = pd.DataFrame(results)
            
            if df.empty:
                self.logger.warning(f"No results collected for year {year}")
            else:
                self.logger.info(f"Successfully collected {len(df)} results for year {year}")

            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching race results for {year}: {str(e)}")
            return pd.DataFrame()

    def get_available_grand_prix(self) -> List[str]:
        """Get list of all available Grand Prix names."""
        return list(self.GP_NAME_MAPPING.keys())

    def get_next_race(self) -> Dict[str, Any]:
        """Get information about the next race."""
        try:
            data = self._make_request('current/next')
            race = data['MRData']['RaceTable']['Races'][0]
            
            return {
                'name': next((k for k, v in self.GP_NAME_MAPPING.items() 
                            if v == race['raceName']), race['raceName']),
                'circuit': race['Circuit']['circuitName'],
                'date': datetime.strptime(f"{race['date']} {race['time']}", "%Y-%m-%d %H:%M:%SZ"),
                'round': int(race['round']),
                'location': {
                    'locality': race['Circuit']['Location']['locality'],
                    'country': race['Circuit']['Location']['country'],
                    'lat': float(race['Circuit']['Location']['lat']),
                    'long': float(race['Circuit']['Location']['long'])
                }
            }
        except Exception as e:
            self.logger.error(f"Error fetching next race: {str(e)}")
            # Return fallback data if API fails
            return {
                'name': "Next GP",
                'circuit': "Unknown Circuit",
                'date': datetime.now() + timedelta(days=14),
                'round': 0,
                'location': {
                    'locality': "Unknown",
                    'country': "Unknown",
                    'lat': 0.0,
                    'long': 0.0
                }
            }

    def get_driver_standings(self, year: Optional[int] = None) -> pd.DataFrame:
        """Get driver standings for a specific year."""
        endpoint = f"{year}/driverStandings" if year else "current/driverStandings"
        
        try:
            data = self._make_request(endpoint)
            standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
            
            results = []
            for standing in standings:
                results.append({
                    'position': int(standing['position']),
                    'driver': f"{standing['Driver']['givenName']} {standing['Driver']['familyName']}",
                    'driver_id': standing['Driver']['driverId'],
                    'points': float(standing['points']),
                    'wins': int(standing['wins']),
                    'constructor': standing['Constructors'][0]['name'],
                    'nationality': standing['Driver']['nationality']
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error fetching driver standings: {str(e)}")
            return pd.DataFrame()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob('*.json'):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")