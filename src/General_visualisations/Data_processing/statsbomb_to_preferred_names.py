"""
Extract and match player names from StatsBomb data with Wikidata preferred names.

This combines:
1. Existing StatsBomb event data
2. Wikidata for preferred player names
3. Fuzzy matching to connect them
"""

import pandas as pd
from fuzzywuzzy import fuzz
import logging

# Define logger for tracking progress and issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Catch any import errors for the wikidata_scrape module
try:
    from .wikidata_scrape import get_players_by_names
except ImportError:
    from Data_processing.wikidata_scrape import get_players_by_names


def extract_statsbomb_players(events_df, team_name):
    """
    Extract unique player names from StatsBomb event data for a specific team.
    
    Args:
        events_df: StatsBomb events DataFrame
        team_name: Name of the team
    
    Returns:
        List of unique player names
    """
    team_events = events_df[events_df['team_name'] == team_name]
    player_names = sorted(team_events['player_name'].dropna().unique().tolist())
    
    logger.info(f"Found {len(player_names)} unique players from StatsBomb for {team_name}")
    return player_names


def get_preferred_names_from_wikidata(team_name, player_names):
    """Look up the StatsBomb players for one team in Wikidata."""
    logger.info(f"Fetching preferred names from Wikidata for {team_name}...")
    return get_players_by_names(player_names, country_name=team_name)


def match_statsbomb_to_wikidata(statsbomb_players, wikidata_df, threshold=75):
    """
    Match StatsBomb player names to Wikidata preferred names.
    
    Args:
        statsbomb_players: List of player names from StatsBomb
        wikidata_df: DataFrame with Wikidata player info
        threshold: Minimum confidence score to accept match
    
    Returns:
        DataFrame with matches
    """
    matches = []
    for sb_player in statsbomb_players:
        player_rows = wikidata_df[wikidata_df['statsbomb_name'].eq(sb_player)]

        if player_rows.empty:
            matches.append({
                'statsbomb_name': sb_player,
                'wikidata_preferred_name': None,
                'confidence': 0,
                'match_status': 'NO MATCH',
            })
            continue

        wd_player = player_rows.iloc[0]
        best_match = wd_player['preferred_name']
        best_score = fuzz.token_sort_ratio(sb_player.lower(), best_match.lower())
        
        if best_score >= threshold:
            status = 'MATCHED'
        else:
            status = 'LOW CONFIDENCE' if best_match else 'NO MATCH'
        
        match_info = {
            'statsbomb_name': sb_player,
            'wikidata_preferred_name': best_match,
            'confidence': best_score,
            'match_status': status,
        }
        
        # Add Wikidata info if available
        if best_score >= 70:
            match_info['position'] = wd_player.get('position', '')
            match_info['birth_date'] = wd_player.get('birth_date', '')
            match_info['wikidata_id'] = wd_player.get('wikidata_id', '')
        
        matches.append(match_info)
    
    return pd.DataFrame(matches)


def create_player_name_mapping(events_df, team_name, output_path=None):
    """
    Complete workflow: Extract StatsBomb names, get Wikidata preferred names, match them.
    
    Args:
        events_df: StatsBomb events DataFrame
        team_name: Name of the team/country
        output_path: Optional path to save CSV
    
    Returns:
        DataFrame with matched players
    """
    print(f"\n{'='*60}")
    print(f"Creating player name mapping for {team_name}")
    print(f"{'='*60}")
    
    # Step 1: Extract StatsBomb players
    print(f"\n--- Step 1: Extract {team_name} players from StatsBomb ---")
    sb_players = extract_statsbomb_players(events_df, team_name)
    print(f"  Extracted {len(sb_players)} players:")
    for i, player in enumerate(sb_players[:5], 1):
        print(f"  {i}. {player}")
    if len(sb_players) > 5:
        print(f"  ... and {len(sb_players) - 5} more")
    
    # Step 2: Get Wikidata preferred names
    print(f"\n--- Step 2: Fetch preferred names from Wikidata ---")
    wikidata_df = get_preferred_names_from_wikidata(team_name, sb_players)
    
    if wikidata_df.empty:
        logger.error(f"No Wikidata players found for {team_name}")
        return pd.DataFrame()
    
    print(f"  Found {len(wikidata_df)} players on Wikidata")
    
    # Step 3: Fuzzy match them
    print(f"\n--- Step 3: Fuzzy match StatsBomb names to Wikidata ---")
    matched_df = match_statsbomb_to_wikidata(sb_players, wikidata_df, threshold=75)
    
    # Print results
    print("\n  Match Results:")
    print(matched_df[['statsbomb_name', 'wikidata_preferred_name', 'confidence', 'match_status']].to_string())
    
    # Statistics
    matched_count = (matched_df['match_status'] == 'MATCHED').sum()
    low_conf = (matched_df['match_status'] == 'LOW CONFIDENCE').sum()
    no_match = (matched_df['match_status'] == 'NO MATCH').sum()
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"    Matched: {matched_count}/{len(matched_df)}")
    print(f"    Low Confidence: {low_conf}/{len(matched_df)}")
    print(f"    No Match: {no_match}/{len(matched_df)}")
    print(f"{'='*60}\n")
    
    # Save if requested
    if output_path:
        matched_df.to_csv(output_path, index=False)
        logger.info(f"  Saved to {output_path}")
    
    return matched_df


