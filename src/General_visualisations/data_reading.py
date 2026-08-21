import ast
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mplsoccer import Pitch
import requests
from ipywidgets import Dropdown, interact
from IPython.display import display, HTML
# Global variables
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
PITCH_X, PITCH_Y = 120, 80



def read_json(url: str):
    """Read JSON from a URL (slightly more robust than pd.read_json)."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def load_competitions():
    url = f"{BASE}/competitions.json"
    comps = pd.DataFrame(read_json(url))
    return comps


def load_matches(competition_id: int, season_id: int):
    url = f"{BASE}/matches/{competition_id}/{season_id}.json"
    matches = pd.DataFrame(read_json(url))
    return matches


def load_events(match_id: int):
    url = f"{BASE}/events/{match_id}.json"
    events = read_json(url)

    # Flatten nested dicts into columns like type.name, team.name, etc.
    df = pd.json_normalize(events, sep=".")
    df["match_id"] = match_id
    return df

def pick_season_competition(comps: pd.DataFrame, preferred_name=None, preferred_season="2015/2016"):
    # Check if the competition takes place over two years
    if "/" in preferred_season:
        
        # Select competition for a given season
        first_year = preferred_season.split("/")[0]
        second_year = preferred_season.split("/")[1]
        
        # season_name varies => catches the required season
        mask = comps["season_name"].astype(str).str.contains(first_year, na=False) & comps["season_name"].astype(str).str.contains(second_year, na=False)
    
    else:
        year = preferred_season
        mask = comps["season_name"].astype(str).str.contains(year, na=False)
        
    c = comps[mask].copy()

    if preferred_name is not None:
        c2 = c[c["competition_name"].eq(preferred_name)]
        if len(c2):
            return c2.iloc[0]

    if len(c) == 0:
        raise ValueError("No season found in competitions.json for this open dataset.")
    return c.iloc[0]


def _parse_lineup_value(lineup_value):
    """Parse StatsBomb lineup data making sure to catch it even if the type varies."""
    # Reformat into a list type
    if isinstance(lineup_value, list):
        return lineup_value
    if isinstance(lineup_value, dict):
        return [lineup_value]
    if not isinstance(lineup_value, str):
        return []

    # Perform checks for str types
    value = lineup_value.strip()
    if not value:
        return []
    # Try to interpret as a python string
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Try to interpret as a JSON string 
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []

    # Reformat into a list type
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def get_player_positions(events: pd.DataFrame, team_name: str, season_level: bool = False):
    """
    Extract a team's starting XI positions from StatsBomb lineup data.

    If season_level=True, it finds the modal position for each player.
    """
    if events.empty:
        return {}

    type_col = "type.name" if "type.name" in events.columns else "event_type"
    team_col = "team.name" if "team.name" in events.columns else "team_name"
    lineup_col = "tactics.lineup"

    # Decide if to use season-level position
    if season_level:
        if "match_id" not in events.columns:
            return {}

        positions = {}

        for _, match_events in events.groupby("match_id", dropna=False):
            xi_rows = match_events[(match_events[type_col] == "Starting XI") & (match_events[team_col] == team_name)]
            if xi_rows.empty:
                continue

            lineup = _parse_lineup_value(xi_rows.iloc[0].get(lineup_col, []))

            for p in lineup:
                if not isinstance(p, dict):
                    continue

                player_info = p.get("player", {})
                position_info = p.get("position", {})
                player_name = player_info.get("name")
                pos_name = position_info.get("name")

                if player_name and pos_name:
                    if player_name not in positions:
                        positions[player_name] = {}
                    if pos_name not in positions[player_name]:
                        positions[player_name][pos_name] = 0
                    positions[player_name][pos_name] += 1
        return {
            player_name: max(counts, key=counts.get)
            for player_name, counts in positions.items()
        }
        # position_counts = defaultdict(Counter)
        # for _, match_events in events.groupby("match_id", dropna=False):
        #     xi_rows = match_events[(match_events[type_col] == "Starting XI") & (match_events[team_col] == team_name)]
        #     if xi_rows.empty:
        #         continue

        #     lineup = _parse_lineup_value(xi_rows.iloc[0].get(lineup_col, []))
        #     for p in lineup:
        #         if not isinstance(p, dict):
        #             continue
        #         player_info = p.get("player", {})
        #         position_info = p.get("position", {})
        #         player_name = player_info.get("name")
        #         pos_name = position_info.get("name")
        #         if player_name and pos_name:
        #             position_counts[player_name][pos_name] += 1

        # return {
        #     player_name: counter.most_common(1)[0][0]
        #     for player_name, counter in position_counts.items()
        # }

    xi_rows = events[(events[type_col] == "Starting XI") & (events[team_col] == team_name)]
    if xi_rows.empty:
        return {}

    lineup = _parse_lineup_value(xi_rows.iloc[0].get(lineup_col, []))
    pos_map = {}
    for p in lineup:
        if not isinstance(p, dict):
            continue

        player_info = p.get("player", {})
        position_info = p.get("position", {})
        player_name = player_info.get("name")
        pos_name = position_info.get("name")
        if player_name and pos_name:
            pos_map[player_name] = pos_name

    return pos_map


def load_all_season_data(competition_id: int, season_id: int, sleep: float = 0.1) -> dict:
    """
    Load ALL match events for a season ONCE.
    Returns dict with all data needed for any analysis.
    """
    matches = load_matches(competition_id, season_id)
    
    # Get all teams
    teams = set()
    for _, m in matches.iterrows():
        teams.add(m["home_team"]["home_team_name"])
        teams.add(m["away_team"]["away_team_name"])
    teams = sorted(list(teams))
    
    print(f"Loading {len(matches)} matches for {len(teams)} teams...")
    
    # Load all events ONCE
    all_events = {}
    for i, mid in enumerate(matches["match_id"].unique()):
        if i % 10 == 0:
            print(f"  Loading match {i+1}/{len(matches)}...")
        all_events[int(mid)] = load_events(int(mid))
        time.sleep(sleep)
    
    print("Data loaded")
    
    return {
        "matches": matches,
        "teams": teams,
        "events": all_events,  # All events stored in a dict keyed by match_id
    }


def build_events_df(all_events: dict, matches: pd.DataFrame) -> pd.DataFrame:
    """Concatenate match event tables into one season-level DataFrame."""
    events_df = pd.concat(all_events.values(), ignore_index=True)

    # Add x,y coords safely, even when some location values are missing or incorrectly formatted
    if "location" in events_df.columns:
        location_values = [
            loc if isinstance(loc, (list, tuple)) and len(loc) == 2 else [np.nan, np.nan]
            for loc in events_df["location"].tolist()
        ]
        coords = pd.DataFrame(location_values, index=events_df.index, columns=["x", "y"])
        events_df = pd.concat([events_df, coords], axis=1)

    # Rename columns for easier access
    events_df["event_type"] = events_df["type.name"]
    events_df["team_name"] = events_df["team.name"]
    events_df["player_name"] = events_df["player.name"]
    events_df["match_date"] = events_df["match_id"].map(matches.set_index("match_id")["match_date"])

    match_index = matches.set_index("match_id")
    home_team_names = match_index["home_team"].apply(lambda d: d.get("home_team_name") if isinstance(d, dict) else np.nan)
    away_team_names = match_index["away_team"].apply(lambda d: d.get("away_team_name") if isinstance(d, dict) else np.nan)
    events_df["home_team"] = events_df["match_id"].map(home_team_names)
    events_df["away_team"] = events_df["match_id"].map(away_team_names)

    # Simplify outcome columns if they exist
    if "shot.outcome.name" in events_df.columns:
        events_df["shot_outcome"] = events_df["shot.outcome.name"]
    if "pass.outcome.name" in events_df.columns:
        events_df["pass_outcome"] = events_df["pass.outcome.name"]

    return events_df


def get_events(target={"Competition":"FIFA World Cup","Season": "2022"}):
    """ Create event dataframe if not already existing """
    
    # Get file name
    file_name = f"event_data/events_df_{target['Competition']}_{target['Season']}.csv".strip()

    if os.path.exists(file_name):
        events_df = pd.read_csv(file_name, index_col=None)
        print("Event data read from file")
    else:
        comps = load_competitions()

        season = pick_season_competition(comps, preferred_name=target[0], preferred_season=target[1])
        print("Selected season:", season["competition_name"], season["season_name"])

        season_data = load_all_season_data(season["competition_id"], season["season_id"], sleep=0.05)


        print(f"Creating events df for {season['competition_name']} {season['competition_id']}")

        events_df = build_events_df(season_data["events"], season_data["matches"])
        events_df.to_csv(file_name)
    
    print("Events loaded:", events_df.shape)
    return events_df