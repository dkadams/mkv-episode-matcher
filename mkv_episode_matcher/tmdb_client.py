# tmdb_client.py
from functools import cache

import requests
from loguru import logger

BASE_IMAGE_URL = "https://image.tmdb.org/t/p/original"

def search_series(config, series_name, page=1):
    """
    Search TMDB for a series by name.

    Args:
        config (Config): Configuration object
        series_name (str): The name of the series.
        page (int): The page of data to fetch
    Returns:
        Any: JSON results of the query
    """
    tmdb_api_key = config.stored.get("api", "tmdb_api_key")
    response = requests.get(f"https://api.themoviedb.org/3/search/tv", {
        "query": series_name,
        "api_key": tmdb_api_key,
        "page": page
    })
    response.raise_for_status()
    if response.status_code == 200:
        return response.json()
    else:
        # Don't expect to get here
        return None

def fetch_series_detail(config, series_id):
    """
    Fetch the TMDb data for a series.

    Args:
        config (Config): Configuration object
        series_id (int): A TMDB series id
    Returns:
        Any: Series details JSON
    """
    tmdb_api_key = config.stored.get("api", "tmdb_api_key")
    response = requests.get(f"https://api.themoviedb.org/3/tv/{series_id}", {
        "api_key": tmdb_api_key
    })
    response.raise_for_status()
    if response.status_code == 200:
        return response.json()
    else:
        # Don't expect to get here
        return None

def fetch_season_details(config, show_id, season_numbers):
    """
    Fetches the series details and the details for the specified seasons.

    Args:
        config (Config): Configuration object
        show_id (str): The ID of the show on TMDb.
        season_numbers (list[int]): The season numbers to fetch details for.

    Returns:
        int: The total number of episodes in the season, or 0 if the API request failed.
    """
    logger.info(f"Fetching season details for Season {season_numbers}...")
    tmdb_api_key = config.stored.get("api", "tmdb_api_key")
    try:
        season_reqs = [f"season/{season_number}" for season_number in season_numbers]
        response = requests.get(f"https://api.themoviedb.org/3/tv/{show_id}", {
            "api_key": tmdb_api_key,
            "append_to_response": ",".join(season_reqs)
        })
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch season details for Season {season_numbers}: {e}")
        return 0
    except KeyError:
        logger.error(
            f"Missing 'episodes' key in response JSON data for Season {season_numbers}"
        )
        return None


def get_number_of_seasons(show_id):
    """
    Retrieves the number of seasons for a given TV show from the TMDB API.

    Parameters:
    - show_id (int): The ID of the TV show.

    Returns:
    - num_seasons (int): The number of seasons for the TV show.

    Raises:
    - requests.HTTPError: If there is an error while making the API request.
    """
    config = _get_config(CONFIG_FILE)
    tmdb_api_key = config.get("tmdb_api_key")
    url = f"https://api.themoviedb.org/3/tv/{show_id}?api_key={tmdb_api_key}"
    response = requests.get(url)
    response.raise_for_status()
    show_data = response.json()
    num_seasons = show_data.get("number_of_seasons", 0)
    logger.info(f"Found {num_seasons} seasons")
    return num_seasons
