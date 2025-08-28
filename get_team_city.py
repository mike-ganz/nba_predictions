def get_team_city(team_name):
    """
    Converts a full NBA team name to just the city/state name
    
    Args:
        team_name (str): Full team name
        
    Returns:
        str: City/state portion of team name
    """
    # Dictionary mapping full names to city/state names
    team_cities = {
        'Atlanta Hawks': 'Atlanta',
        'Boston Celtics': 'Boston', 
        'Brooklyn Nets': 'Brooklyn',
        'Charlotte Hornets': 'Charlotte',
        'Chicago Bulls': 'Chicago',
        'Cleveland Cavaliers': 'Cleveland',
        'Dallas Mavericks': 'Dallas',
        'Denver Nuggets': 'Denver',
        'Detroit Pistons': 'Detroit',
        'Golden State Warriors': 'Golden State',
        'Houston Rockets': 'Houston',
        'Indiana Pacers': 'Indiana',
        'Los Angeles Clippers': 'Los Angeles',
        'Los Angeles Lakers': 'Los Angeles',
        'Memphis Grizzlies': 'Memphis',
        'Miami Heat': 'Miami',
        'Milwaukee Bucks': 'Milwaukee',
        'Minnesota Timberwolves': 'Minnesota',
        'New Orleans Pelicans': 'New Orleans',
        'New York Knicks': 'New York',
        'Oklahoma City Thunder': 'Oklahoma City',
        'Orlando Magic': 'Orlando',
        'Philadelphia 76ers': 'Philadelphia',
        'Phoenix Suns': 'Phoenix',
        'Portland Trail Blazers': 'Portland',
        'Sacramento Kings': 'Sacramento',
        'San Antonio Spurs': 'San Antonio',
        'Toronto Raptors': 'Toronto',
        'Utah Jazz': 'Utah',
        'Washington Wizards': 'Washington'
    }
    
    return team_cities.get(team_name, team_name)

def get_team_city_LA(team_name):
    """
    Converts a full NBA team name to just the city/state name
    
    Args:
        team_name (str): Full team name
        
    Returns:
        str: City/state portion of team name
    """
    # Dictionary mapping full names to city/state names
    team_cities = {
        'Atlanta Hawks': 'Atlanta',
        'Boston Celtics': 'Boston', 
        'Brooklyn Nets': 'Brooklyn',
        'Charlotte Hornets': 'Charlotte',
        'Chicago Bulls': 'Chicago',
        'Cleveland Cavaliers': 'Cleveland',
        'Dallas Mavericks': 'Dallas',
        'Denver Nuggets': 'Denver',
        'Detroit Pistons': 'Detroit',
        'Golden State Warriors': 'Golden State',
        'Houston Rockets': 'Houston',
        'Indiana Pacers': 'Indiana',
        'Los Angeles Clippers': 'LA Clippers',
        'Los Angeles Lakers': 'LA Lakers',
        'Memphis Grizzlies': 'Memphis',
        'Miami Heat': 'Miami',
        'Milwaukee Bucks': 'Milwaukee',
        'Minnesota Timberwolves': 'Minnesota',
        'New Orleans Pelicans': 'New Orleans',
        'New York Knicks': 'New York',
        'Oklahoma City Thunder': 'Oklahoma City',
        'Orlando Magic': 'Orlando',
        'Philadelphia 76ers': 'Philadelphia',
        'Phoenix Suns': 'Phoenix',
        'Portland Trail Blazers': 'Portland',
        'Sacramento Kings': 'Sacramento',
        'San Antonio Spurs': 'San Antonio',
        'Toronto Raptors': 'Toronto',
        'Utah Jazz': 'Utah',
        'Washington Wizards': 'Washington'
    }
    
    return team_cities.get(team_name, team_name)

def get_full_team_name(city):
    """
    Converts a city/state name to the full NBA team name
    
    Args:
        city (str): City/state name
        
    Returns:
        str: Full team name, or original city name if not found
    """
    # Dictionary mapping city/state names to full team names
    city_teams = {
        'Atlanta': 'Atlanta Hawks',
        'Boston': 'Boston Celtics',
        'Brooklyn': 'Brooklyn Nets', 
        'Charlotte': 'Charlotte Hornets',
        'Chicago': 'Chicago Bulls',
        'Cleveland': 'Cleveland Cavaliers',
        'Dallas': 'Dallas Mavericks',
        'Denver': 'Denver Nuggets',
        'Detroit': 'Detroit Pistons',
        'Golden State': 'Golden State Warriors',
        'Houston': 'Houston Rockets',
        'Indiana': 'Indiana Pacers',
        'Los Angeles': 'Los Angeles Lakers',  # Note: Returns Lakers by default for LA
        'Memphis': 'Memphis Grizzlies',
        'Miami': 'Miami Heat',
        'Milwaukee': 'Milwaukee Bucks',
        'Minnesota': 'Minnesota Timberwolves',
        'New Orleans': 'New Orleans Pelicans',
        'New York': 'New York Knicks',
        'Oklahoma City': 'Oklahoma City Thunder',
        'Orlando': 'Orlando Magic',
        'Philadelphia': 'Philadelphia 76ers',
        'Phoenix': 'Phoenix Suns',
        'Portland': 'Portland Trail Blazers',
        'Sacramento': 'Sacramento Kings',
        'San Antonio': 'San Antonio Spurs',
        'Toronto': 'Toronto Raptors',
        'Utah': 'Utah Jazz',
        'Washington': 'Washington Wizards'
    }
    
    return city_teams.get(city, city)


def get_team_abbreviation(team_name):
    """
    Converts a full team name to its 3-letter abbreviation
    
    Args:
        team_name (str): Full team name
        
    Returns:
        str: 3-letter team abbreviation, or original name if not found
    """
    team_abbrev = {
        'Atlanta Hawks': 'ATL',
        'Boston Celtics': 'BOS', 
        'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA',
        'Chicago Bulls': 'CHI',
        'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL',
        'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW',
        'Houston Rockets': 'HOU',
        'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NO',
        'New York Knicks': 'NYK',
        'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI',
        'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC',
        'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTH',
        'Washington Wizards': 'WAS'
    }
    
    return team_abbrev.get(team_name, team_name)
