import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mplsoccer import Pitch
from collections import defaultdict
import os

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
PITCH_X, PITCH_Y = 120, 80


def read_json(url: str):
    """Read JSON from a URL (slightly more robust than pd.read_json)."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def season_shots_df(matches):
    match_ids = matches["match_id"].tolist()
    parts = []
    for i, mid in enumerate(match_ids, start=1):
        print(f"Match {i}/{len(match_ids)}: {mid}")
        try:
            parts.append(load_shots(mid))
        except Exception as e:
            print(f"Failed match_id={mid}: {e}")

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def store_data_locally(matches, path="Data/shot_data.csv"):

    shots = season_shots_df(matches)
    # optional: keep only columns you care about (smaller CSV, faster load)
    keep = [c for c in [
        "match_id", "team.name", "player.name",
        "minute", "second",
        "x", "y", "xg", "is_goal",
        "shot.outcome.name", "shot.body_part.name", "shot.technique.name",
        "shot.type.name"
    ] if c in shots.columns]
    shots = shots[keep]

    shots.to_csv(path, index=False)

def load_competitions():
    url = f"{BASE}/competitions.json"
    comps = pd.DataFrame(read_json(url))
    return comps



def load_shots(match_id: int) -> pd.DataFrame:
    url = f"{BASE}/events/{match_id}.json"
    events = read_json(url)

    shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]
    df = pd.json_normalize(shots, sep=".")
    df["match_id"] = match_id

    # coords
    loc = df["location"].apply(lambda v: v if isinstance(v, list) and len(v) >= 2 else [np.nan, np.nan])
    df["x"] = loc.apply(lambda v: v[0])
    df["y"] = loc.apply(lambda v: v[1])

    df["xg"] = df.get("shot.statsbomb_xg", 0.0).fillna(0.0)
    df["is_goal"] = df.get("shot.outcome.name", "").eq("Goal")

    return df.dropna(subset=["x", "y"])


def load_matches(competition_id: int, season_id: int):
    url = f"{BASE}/matches/{competition_id}/{season_id}.json"
    matches = pd.DataFrame(read_json(url))
    return matches


def load_events(match_id: int):
    url = f"{BASE}/events/{match_id}.json"
    events = read_json(url)
    shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]

    # Flatten nested dicts into columns like type.name, team.name, etc.
    df = pd.json_normalize(shots, sep=".")
    df["match_id"] = match_id
    return df

def get_matches_season(chosen_season=['Premier League','2015/2016']):
    # Browse competitions
    comps = load_competitions()

    # Pick a competition and season
    chosen_season = ['Premier League','2015/2016']
    comp_row = pick_season_competition(comps, preferred_name=chosen_season[0], preferred_season=chosen_season[1])
    competition_id = int(comp_row["competition_id"])
    season_id = int(comp_row["season_id"])

    print("Using:", comp_row[["competition_name", "season_name", "country_name", "competition_id", "season_id"]].to_dict())
    # Extract matches
    matches = load_matches(competition_id, season_id)
    
    return matches, chosen_season


def add_location_xy(df: pd.DataFrame):
    """Split StatsBomb 'location' [x,y] into columns."""
    if "location" in df.columns:
        df["x"] = df["location"].apply(lambda v: v[0] if isinstance(v, list) and len(v) == 2 else None)
        df["y"] = df["location"].apply(lambda v: v[1] if isinstance(v, list) and len(v) == 2 else None)
    return df



def add_xy(df: pd.DataFrame):
    # start location
    df["x"] = df["location"].apply(lambda v: v[0] if isinstance(v, list) and len(v) >= 2 else np.nan)
    df["y"] = df["location"].apply(lambda v: v[1] if isinstance(v, list) and len(v) >= 2 else np.nan)
    # pass end location
    if "pass.end_location" in df.columns:
        df["end_x"] = df["pass.end_location"].apply(lambda v: v[0] if isinstance(v, list) and len(v) >= 2 else np.nan)
        df["end_y"] = df["pass.end_location"].apply(lambda v: v[1] if isinstance(v, list) and len(v) >= 2 else np.nan)
    return df



def plot_event_type_counts(events: pd.DataFrame, top_n=15):
    plt.figure(figsize=(10, 5))
    counts = events["type.name"].value_counts().head(top_n)
    sns.barplot(x=counts.values, y=counts.index, color="steelblue")
    plt.title(f"Top {top_n} event types (match_id={events['match_id'].iloc[0]})")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def pick_season_competition(comps: pd.DataFrame, preferred_name=None, preferred_season="2015/2016"):
    # Select competition for a given season (and optionally name)
    first_year = preferred_season.split("/")[0]
    second_year = preferred_season.split("/")[1]
    
    # season_name varies; this catches the required season
    mask = comps["season_name"].astype(str).str.contains(first_year, na=False) & comps["season_name"].astype(str).str.contains(second_year, na=False)
    c = comps[mask].copy()

    if preferred_name is not None:
        c2 = c[c["competition_name"].eq(preferred_name)]
        if len(c2):
            return c2.iloc[0]

    if len(c) == 0:
        raise ValueError("No 2015/16 season found in competitions.json for this open dataset.")
    return c.iloc[0]


def get_game_shots(events,team_name=None):
    # Get shots
    shots = events[events["type.name"] == "Shot"].copy()
    shots = add_location_xy(shots)

    # Optional: filter by team
    if team_name is not None:
        shots = shots[shots["team.name"] == team_name]

    return shots



def get_season_shots(season_matches, season, team_name=None):
    match_shots = []
    # Loop through every match 
    for i, m in season_matches.iterrows():
        m.to_dict()
        # Get playing teams for the match
        playing_teams = [m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]]
        
        # Only get shots for matches involving the specified team (or all matches if team_name is None)
        if team_name is None or team_name in playing_teams:
            print(f'Match: {i}/{len(season_matches.index)}')
            match_id = m['match_id']
            events = load_events(match_id)
            shots = get_game_shots(events, team_name=team_name)
            match_shots.append(shots)

    return pd.concat(match_shots, ignore_index=True) if match_shots else pd.DataFrame()

def get_game_shot_map(events, team_name=None):
    # Get shots
    shots = get_game_shots(events, team_name=team_name)

    # Some matches/competitions may have missing xG
    if "shot.statsbomb_xg" in shots.columns:
        shots["xg"] = shots["shot.statsbomb_xg"].fillna(0.0)
    else:
        shots["xg"] = 0.0

    # Identify goals
    # In StatsBomb, shot outcome is in shot.outcome.name (e.g. "Goal")
    shots["is_goal"] = shots.get("shot.outcome.name", "").eq("Goal")

    # Drop rows without coordinates
    shots = shots.dropna(subset=["x", "y"])

    # Plot non-goals then goals on top
    ng = shots[~shots["is_goal"]]
    g = shots[shots["is_goal"]]

    return g, ng, shots['xg']

def get_season_shot_map(season_matches, season, team_name=None):
    
    match_shots = []
    goals, no_goals, xgs = [], [], []
    # Loop through every match 
    for i, m in season_matches.iterrows():
        m.to_dict()
        playing_teams = [m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]]
        if team_name is None:
            print(f'Match: {i}/{len(season_matches.index)}')
            match_id = m['match_id']
            events = load_events(match_id)
            g, ng, xg = get_game_shot_map(events)
            goals.append(g)
            no_goals.append(ng)
            xgs.append(xg)

            g_all = pd.concat(goals, ignore_index=True) if goals else pd.DataFrame()
            ng_all = pd.concat(no_goals, ignore_index=True) if no_goals else pd.DataFrame()
            xg_all = pd.concat(xgs, ignore_index=True) if xgs else pd.Series(dtype=float)
            
        elif team_name in playing_teams:
            print(f'Match: {i}/{len(season_matches.index)}')
            match_id = m['match_id']
            events = load_events(match_id)
            g, ng, xg = get_game_shot_map(events, team_name=team_name)
            goals.append(g)
            no_goals.append(ng)
            xgs.append(xg)

            # match_shots.append([g, ng, shots['xg']])
            # match_dict[match_id] = {'goals':g, 'no_goal': ng, 'xG':shots['xg']}
            g_all = pd.concat(goals, ignore_index=True) if goals else pd.DataFrame()
            ng_all = pd.concat(no_goals, ignore_index=True) if no_goals else pd.DataFrame()
            xg_all = pd.concat(xgs, ignore_index=True) if xgs else pd.Series(dtype=float)
            
    
    return g_all, ng_all, xg_all



def plot_shot_map(g, ng, xg, team_name=None, match_id=None):

    pitch = Pitch(pitch_type="statsbomb", line_color="#222222")
    fig, ax = pitch.draw(figsize=(10, 7))

    # Marker size scaled by xG
    sizes = 100 + xg * 1200

    pitch.scatter(ng["x"], ng["y"], s=sizes.loc[ng.index], c="#1f77b4", alpha=0.6, ax=ax, label="Shot (no goal)")
    pitch.scatter(g["x"], g["y"], s=sizes.loc[g.index], c="#d62728", alpha=0.7, ax=ax, label="Goal")

    if match_id != None and team_name != None:
        title = f"Shot map (match_id={match_id})"
    else:
        title = f"Shot map"

    if team_name:
        title += f" — {team_name}"
    ax.set_title(title)
    ax.legend(loc="upper left")
    plt.show()

def event_time_seconds(df: pd.DataFrame) -> pd.Series:
    return df["minute"].fillna(0).astype(float) * 60 + df["second"].fillna(0).astype(float)


def outside_box_shots(shots_df):
    # Define penalty box dimensions (StatsBomb coordinates)
    box_x_min, box_x_max = 0, 120
    box_y_min, box_y_max = 18, 62

    # Check if shots are outside the box
    shots_df["is_outside_box"] = ~(
        (shots_df["x"] >= box_x_min) &
        (shots_df["x"] <= box_x_max) &
        (shots_df["y"] >= box_y_min) &
        (shots_df["y"] <= box_y_max)
    )

    # Shot percentage outside the box
    total_shots = len(shots_df)
    outside_box_shots = shots_df["is_outside_box"].sum()
    percentage_outside = (outside_box_shots / total_shots) * 100 if total_shots > 0 else 0


    return shots_df, percentage_outside




def map_comparison(matches, season, team_names, plot_maps=False):
    '''Compare shot maps for multiple teams in the same season'''
    g_list, ng_list, xg_list = [], [], []

    for team in team_names:
        # Get shot map for whole season
        print(f"\nGetting shot map for {team} in season {season[1]}...")
        g, ng, xg = get_season_shot_map(matches, season=season, team_name=team)
        
        g_list.append(g)
        ng_list.append(ng)
        xg_list.append(xg)
    
        # Plot shot map
        if plot_maps:
            plot_shot_map(g, ng, xg, team_name=team)

        ################## Incomplete ##################
    

def main_1():
    # Browse competitions
    comps = load_competitions()
    
    # Pick a competition and season
    comp_row = pick_season_competition(comps, preferred_name='Premier League', preferred_season="2015/2016")
    competition_id = int(comp_row["competition_id"])
    season_id = int(comp_row["season_id"])

    print("Using:", comp_row[["competition_name", "season_name", "country_name", "competition_id", "season_id"]].to_dict())

    matches = load_matches(competition_id, season_id)
   
    # Choose a random match from the competition
    rand_idx = np.random.choice(len(matches), size=1, replace=False)
    match = matches.iloc[rand_idx[0]]
    
    # Get events for the chosen match
    events = load_events(match["match_id"])

    # Basic visuals
    # plot_event_type_counts(events, top_n=15)

    # Get shot map for game
    g, ng, xg = get_game_shot_map(events)
    
    # Shot map for whole match
    plot_shot_map(g, ng, xg)


    # Shot map for just one team (uncomment and set team name)
    # team = events["team.name"].dropna().unique()[0]
    # plot_shot_map(events, team_name=team)



def main_2():

    # Get matches for season
    matches, chosen_season = get_matches_season(chosen_season=['Premier League','2015/2016'])


    # 'Swansea City', 'West Ham United', 'Everton', 'Southampton', 'Liverpool',
    # 'Newcastle United', 'Crystal Palace', 'Arsenal', 'Manchester United',
    # 'Tottenham Hotspur', 'Stoke City', 'Chelsea', 'Aston Villa', 'Norwich City',
    # 'Watford', 'AFC Bournemouth', 'West Bromwich Albion', 'Leicester City',
    # 'Sunderland', 'Manchester City'


    # Optional: Choose a team to filter by 
    team_name = 'Swansea City'  # set to None for all teams, or pick from printed list above
    # Get shot map for whole season
    print(f"\nGetting shot map for {team_name} in season {chosen_season[1]}...")
    g, ng, xg = get_season_shot_map(matches, season=chosen_season, team_name=team_name)

    print(f"\\Plotting shot map for {team_name}")

    # Plot shot map 
    plot_shot_map(g, ng, xg, team_name=team_name)

def comparison_main():
    # Get matches for season
    matches, chosen_season = get_matches_season(chosen_season=['Premier League','2015/2016'])
    team_names = ['Swansea City', 'Liverpool', 'Manchester City']
    map_comparison(matches, season=chosen_season, team_names=team_names)


if __name__ == "__main__":
    # main_2()
    comparison_main()