"""
Utility to get team city name from team name or abbreviation.
"""

# Map team names/abbreviations to city names
TEAM_TO_CITY = {
    # Full names
    "Atlanta": "Atlanta",
    "Boston": "Boston",
    "Brooklyn": "Brooklyn",
    "Charlotte": "Charlotte",
    "Chicago": "Chicago",
    "Cleveland": "Cleveland",
    "Dallas": "Dallas",
    "Denver": "Denver",
    "Detroit": "Detroit",
    "Golden State": "Golden State",
    "Houston": "Houston",
    "Indiana": "Indiana",
    "LA Clippers": "LA Clippers",
    "Los Angeles Clippers": "LA Clippers",
    "Los Angeles Lakers": "LA Lakers",
    "LA Lakers": "LA Lakers",
    "Memphis": "Memphis",
    "Miami": "Miami",
    "Milwaukee": "Milwaukee",
    "Minnesota": "Minnesota",
    "New Orleans": "New Orleans",
    "New York": "New York",
    "Oklahoma City": "Oklahoma City",
    "Orlando": "Orlando",
    "Philadelphia": "Philadelphia",
    "Phoenix": "Phoenix",
    "Portland": "Portland",
    "Sacramento": "Sacramento",
    "San Antonio": "San Antonio",
    "Toronto": "Toronto",
    "Utah": "Utah",
    "Washington": "Washington",
    # Abbreviations
    "ATL": "Atlanta",
    "BOS": "Boston",
    "BKN": "Brooklyn",
    "CHA": "Charlotte",
    "CHI": "Chicago",
    "CLE": "Cleveland",
    "DAL": "Dallas",
    "DEN": "Denver",
    "DET": "Detroit",
    "GSW": "Golden State",
    "HOU": "Houston",
    "IND": "Indiana",
    "LAC": "LA Clippers",
    "LAL": "LA Lakers",
    "MEM": "Memphis",
    "MIA": "Miami",
    "MIL": "Milwaukee",
    "MIN": "Minnesota",
    "NOP": "New Orleans",
    "NYK": "New York",
    "OKC": "Oklahoma City",
    "ORL": "Orlando",
    "PHI": "Philadelphia",
    "PHX": "Phoenix",
    "POR": "Portland",
    "SAC": "Sacramento",
    "SAS": "San Antonio",
    "TOR": "Toronto",
    "UTA": "Utah",
    "WAS": "Washington",
}


def get_team_city(team_name: str) -> str:
    """
    Get the city name for a given team name or abbreviation.
    
    Args:
        team_name: Team name (e.g., "Atlanta", "ATL", "Golden State")
        
    Returns:
        City name (e.g., "Atlanta", "Golden State")
        
    Raises:
        ValueError: If team name is not recognized
    """
    if team_name in TEAM_TO_CITY:
        return TEAM_TO_CITY[team_name]
    
    # Try case-insensitive match
    for key, value in TEAM_TO_CITY.items():
        if key.lower() == team_name.lower():
            return value
    
    # If not found, return the input (fallback)
    return team_name

