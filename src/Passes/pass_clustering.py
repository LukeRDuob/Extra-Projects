import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
import time
import requests
import pickle
import os
import hashlib

CACHE_DIR = "./statsbomb_cache"
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
PITCH_X, PITCH_Y = 120, 80


# StatsBomb position IDs mapped to approximate pitch coordinates (attacking right)
# These are based on typical formation positions
STATSBOMB_POSITION_COORDS = {
    1: (10, 40),      # Goalkeeper
    2: (30, 72),      # Right Back
    3: (25, 56),      # Right Center Back
    4: (25, 40),      # Center Back
    5: (25, 24),      # Left Center Back
    6: (30, 8),       # Left Back
    7: (50, 72),      # Right Wing Back
    8: (45, 56),      # Right Defensive Midfield
    9: (40, 40),      # Center Defensive Midfield
    10: (45, 24),     # Left Defensive Midfield
    11: (50, 8),      # Left Wing Back
    12: (55, 72),     # Right Midfield
    13: (55, 56),     # Right Center Midfield
    14: (55, 40),     # Center Midfield
    15: (55, 24),     # Left Center Midfield
    16: (55, 8),      # Left Midfield
    17: (70, 72),     # Right Wing
    18: (70, 56),     # Right Attacking Midfield
    19: (70, 40),     # Center Attacking Midfield
    20: (70, 24),     # Left Attacking Midfield
    21: (70, 8),      # Left Wing
    22: (85, 56),     # Right Center Forward
    23: (90, 40),     # Striker / Center Forward
    24: (85, 24),     # Left Center Forward
    25: (85, 40),     # Secondary Striker
}


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



def add_xy(df: pd.DataFrame):
    # start location
    df["x"] = df["location"].apply(lambda v: v[0] if isinstance(v, list) and len(v) >= 2 else np.nan)
    df["y"] = df["location"].apply(lambda v: v[1] if isinstance(v, list) and len(v) >= 2 else np.nan)
    # pass end location
    if "pass.end_location" in df.columns:
        df["end_x"] = df["pass.end_location"].apply(lambda v: v[0] if isinstance(v, list) and len(v) >= 2 else np.nan)
        df["end_y"] = df["pass.end_location"].apply(lambda v: v[1] if isinstance(v, list) and len(v) >= 2 else np.nan)
    return df


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


def event_time_seconds(df: pd.DataFrame) -> pd.Series:
    return df["minute"].fillna(0).astype(float) * 60 + df["second"].fillna(0).astype(float)


def get_starting_xi(events: pd.DataFrame, team_name: str):
    """
    Returns list of (player_id, player_name) from the 'Starting XI' event.
    In json_normalize output, the lineup is typically stored as a list in 'tactics.lineup'.
    """
    xi_rows = events[(events["type.name"] == "Starting XI") & (events["team.name"] == team_name)]
    if xi_rows.empty:
        return []
    lineup = xi_rows.iloc[0].get("tactics.lineup", [])
    if not isinstance(lineup, list):
        return []
    
    
    # Add player ids and names to be returned for the lineup
    out = []
    for p in lineup:
        # p looks like: {'player': {'id':..., 'name':...}, 'position': {...}, ...}
        if isinstance(p, dict) and "player" in p and isinstance(p["player"], dict):
            out.append((p["player"].get("id"), p["player"].get("name")))
    # Remove Nones
    return [(pid, name) for pid, name in out if pid is not None and name is not None]

def minutes_played_from_events(events: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    Approx minutes played using Starting XI + Substitution events.
    Returns DataFrame: player_id, player_name, minutes
    """
    ev = events.copy()
    ev["t_sec"] = event_time_seconds(ev)
    match_end = float(ev["t_sec"].max()) if len(ev) else 0.0

    # Starting XI
    xi = get_starting_xi(ev, team_name)
    start_time = {pid: 0.0 for pid, _ in xi}
    names = {pid: name for pid, name in xi}

    # Substitutions for this team
    subs = ev[(ev["type.name"] == "Substitution") & (ev["team.name"] == team_name)].copy()
    
    # Columns: player.id is player going off; substitution.replacement.id is player coming on
    for _, r in subs.iterrows():
        t = float(r["t_sec"])
        off_id = r.get("player.id")
        off_name = r.get("player.name")
        on_id = r.get("substitution.replacement.id")
        on_name = r.get("substitution.replacement.name")

        # End the player who goes off
        if pd.notna(off_id):
            off_id = int(off_id)
            names.setdefault(off_id, off_name)
            # mark end by setting an end time (to compute later)
            # easiest: store end as match_end and overwrite below
        # start the replacement
        if pd.notna(on_id):
            on_id = int(on_id)
            names.setdefault(on_id, on_name)
            # If player appears multiple times, keep earliest start
            start_time.setdefault(on_id, t)

        # # Store an "off" time
        # if pd.notna(off_id):
        #     # use a separate dict to store end times
        #     pass

    # Set end times (default match end)
    end_time = {pid: match_end for pid in start_time.keys()}

    # Apply off times from substitutions
    for _, r in subs.iterrows():
        t = float(r["t_sec"])
        off_id = r.get("player.id")
        if pd.notna(off_id):
            off_id = int(off_id)
            # Only if we know they had started (XI or subbed in)
            if off_id in end_time:
                end_time[off_id] = min(end_time[off_id], t)

    # build minutes table
    rows = []
    for pid, st in start_time.items():
        et = end_time.get(pid, match_end)
        mins = max(0.0, (float(et) - float(st)) / 60.0)
        rows.append({"player_id": pid, "player_name": names.get(pid, str(pid)), "minutes": mins})

    mins_df = pd.DataFrame(rows).sort_values("minutes", ascending=False).reset_index(drop=True)
    return mins_df


def most_played_11_match(events: pd.DataFrame, team_name: str):
    mins = minutes_played_from_events(events, team_name)
    top11 = mins.head(11)
    return set(top11["player_name"].tolist()), mins


def most_played_11_season(competition_id: int, season_id: int, team_name: str, sleep: float = 0.2):
    matches = load_matches(competition_id, season_id)
    
    mask = matches["home_team"].apply(lambda d: d.get("home_team_name") == team_name) | \
           matches["away_team"].apply(lambda d: d.get("away_team_name") == team_name)
    team_matches = matches[mask]

    total_minutes = defaultdict(float)
    id_to_name = {}

    for mid in team_matches["match_id"].tolist():
        ev = load_events(int(mid))
        mins = minutes_played_from_events(ev, team_name)
        for _, r in mins.iterrows():
            pid = int(r["player_id"])
            total_minutes[pid] += float(r["minutes"])
            id_to_name[pid] = r["player_name"]
        time.sleep(sleep)

    mins_season = pd.DataFrame(
        [{"player_id": pid, "player_name": id_to_name.get(pid, str(pid)), "minutes": m}
         for pid, m in total_minutes.items()]
    ).sort_values("minutes", ascending=False).reset_index(drop=True)

    top11 = mins_season.head(11)
    return set(top11["player_name"].tolist()), mins_season, team_matches



def get_player_positions_from_lineup(events: pd.DataFrame, team_name: str) -> dict:
    """
    Extract player positions from the Starting XI event.
    Returns dict: player_name -> (x, y) based on their formation position.
    """
    xi_rows = events[(events["type.name"] == "Starting XI") & (events["team.name"] == team_name)]
    if xi_rows.empty:
        return {}
    
    lineup = xi_rows.iloc[0].get("tactics.lineup", [])
    if not isinstance(lineup, list):
        return {}
    
    positions = {}
    for p in lineup:
        if isinstance(p, dict) and "player" in p and "position" in p:
            player_name = p["player"].get("name")
            pos_id = p["position"].get("id")
            if player_name and pos_id and pos_id in STATSBOMB_POSITION_COORDS:
                positions[player_name] = STATSBOMB_POSITION_COORDS[pos_id]
    
    return positions


def get_most_common_position_season(competition_id: int, season_id: int, 
                                     team_name: str, team_matches: pd.DataFrame,
                                     sleep: float = 0.1) -> dict:
    """
    For each player, find their most commonly assigned position across all matches.
    Returns dict: player_name -> (x, y)
    """
    from collections import Counter
    
    player_position_counts = defaultdict(Counter)
    
    for mid in team_matches["match_id"].tolist():
        ev = load_events(int(mid))
        xi_rows = ev[(ev["type.name"] == "Starting XI") & (ev["team.name"] == team_name)]
        
        if xi_rows.empty:
            time.sleep(sleep)
            continue
            
        lineup = xi_rows.iloc[0].get("tactics.lineup", [])
        if not isinstance(lineup, list):
            time.sleep(sleep)
            continue
        
        for p in lineup:
            if isinstance(p, dict) and "player" in p and "position" in p:
                player_name = p["player"].get("name")
                pos_id = p["position"].get("id")
                if player_name and pos_id:
                    player_position_counts[player_name][pos_id] += 1
        
        time.sleep(sleep)
    
    # Get most common position for each player
    positions = {}
    for player_name, pos_counter in player_position_counts.items():
        most_common_pos_id = pos_counter.most_common(1)[0][0]
        if most_common_pos_id in STATSBOMB_POSITION_COORDS:
            positions[player_name] = STATSBOMB_POSITION_COORDS[most_common_pos_id]
    
    return positions



def rotate_180(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["x", "y", "end_x", "end_y"]:
        if col in df.columns:
            df[col] = (PITCH_X if "x" in col else PITCH_Y) - df[col]
    return df


def standardize_halves(df: pd.DataFrame) -> pd.DataFrame:
    """Flip period 2/4 so both halves line up. Works if df has x,y and maybe end_x,end_y."""
    p = df.copy()
    flip_periods = {2, 4}
    m = p["period"].isin(flip_periods)

    # flip x,y
    p.loc[m, "x"] = PITCH_X - p.loc[m, "x"]
    p.loc[m, "y"] = PITCH_Y - p.loc[m, "y"]

    # flip end_x,end_y if present
    if "end_x" in p.columns:
        p.loc[m, "end_x"] = PITCH_X - p.loc[m, "end_x"]
    if "end_y" in p.columns:
        p.loc[m, "end_y"] = PITCH_Y - p.loc[m, "end_y"]

    return p


def should_rotate_match_using_shots(events: pd.DataFrame, team_name: str) -> bool:
    ev = add_xy(events.copy())
    shots = ev[(ev["type.name"] == "Shot") & (ev["team.name"] == team_name)].copy()
    if shots.empty:
        return False
    shots = shots.dropna(subset=["x", "y"])
    shots = standardize_halves(shots)
    return shots["x"].median() < (PITCH_X / 2)

def completed_passes(events: pd.DataFrame, team_name: str) -> pd.DataFrame:
    ev = add_xy(events.copy())
    passes = ev[(ev["type.name"] == "Pass") & (ev["team.name"] == team_name)].copy()
    if "pass.outcome.name" in passes.columns:
        passes = passes[passes["pass.outcome.name"].isna()]
    passes = passes.dropna(subset=["player.name", "pass.recipient.name", "x", "y", "end_x", "end_y"])
    set_piece_types = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
    if "pass.type.name" in passes.columns:
        passes = passes[~passes["pass.type.name"].isin(set_piece_types)]
    return passes


def get_cache_path(competition_id: int, season_id: int) -> str:
    """Generate cache file path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"season_{competition_id}_{season_id}.pkl")


def load_all_season_data_cached(competition_id: int, season_id: int,
                                 force_reload: bool = False,
                                 use_async: bool = True,
                                 max_workers: int = 15) -> dict:
    """
    Load season data with local file caching.
    First run: Downloads and saves to disk
    Subsequent runs: Loads from disk instantly
    """
    cache_path = get_cache_path(competition_id, season_id)
    
    # Try loading from cache
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        start = time.time()
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded from cache in {time.time() - start:.2f}s")
        return data
    
    # Load from API
    print("Cache not found, loading from API...")
    
    if use_async:
        data = load_all_season_data_async(competition_id, season_id, max_concurrent=max_workers)
    else:
        data = load_all_season_data_parallel(competition_id, season_id, max_workers=max_workers)
    
    # Save to cache
    print(f"Saving to cache: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    
    return data


def clear_cache(competition_id: int = None, season_id: int = None):
    """Clear cached data."""
    if competition_id and season_id:
        path = get_cache_path(competition_id, season_id)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")
    else:
        # Clear all
        if os.path.exists(CACHE_DIR):
            for f in os.listdir(CACHE_DIR):
                os.remove(os.path.join(CACHE_DIR, f))
            print(f"Cleared all cache files")


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
    
    print("Data loaded!")
    
    return {
        "matches": matches,
        "teams": teams,
        "events": all_events,  # All events stored in memory
    }


def get_team_passes_from_cache(data: dict, team_name: str) -> pd.DataFrame:
    """
    Extract a team's passes from pre-loaded data (no API calls).
    """
    matches = data["matches"]
    all_events = data["events"]
    
    # Find team's matches
    mask = matches["home_team"].apply(lambda d: d.get("home_team_name") == team_name) | \
           matches["away_team"].apply(lambda d: d.get("away_team_name") == team_name)
    team_match_ids = matches[mask]["match_id"].tolist()
    
    # Collect passes from cached events
    all_passes = []
    for mid in team_match_ids:
        ev = all_events[int(mid)]
        p = completed_passes(ev, team_name)
        p = standardize_halves(p)
        if should_rotate_match_using_shots(ev, team_name):
            p = rotate_180(p)
        all_passes.append(p)
    
    if not all_passes:
        return pd.DataFrame()
    
    return pd.concat(all_passes, ignore_index=True)


def get_team_top11_from_cache(data: dict, team_name: str) -> set:
    """
    Get most-played XI from cached data (no API calls).
    """
    matches = data["matches"]
    all_events = data["events"]
    
    mask = matches["home_team"].apply(lambda d: d.get("home_team_name") == team_name) | \
           matches["away_team"].apply(lambda d: d.get("away_team_name") == team_name)
    team_match_ids = matches[mask]["match_id"].tolist()
    
    total_minutes = defaultdict(float)
    id_to_name = {}
    
    for mid in team_match_ids:
        ev = all_events[int(mid)]
        
        # Calculate minutes from this match
        ev_copy = ev.copy()
        ev_copy["t_sec"] = ev_copy["minute"].fillna(0) * 60 + ev_copy["second"].fillna(0)
        match_end = ev_copy["t_sec"].max()
        
        # Get starting XI
        xi_rows = ev[(ev["type.name"] == "Starting XI") & (ev["team.name"] == team_name)]
        if xi_rows.empty:
            continue
        
        lineup = xi_rows.iloc[0].get("tactics.lineup", [])
        if not isinstance(lineup, list):
            continue
        
        for p in lineup:
            if isinstance(p, dict) and "player" in p:
                pid = p["player"].get("id")
                pname = p["player"].get("name")
                if pid and pname:
                    total_minutes[pid] += match_end / 60  # Approximate
                    id_to_name[pid] = pname
    
    # Get top 11 by minutes
    sorted_players = sorted(total_minutes.items(), key=lambda x: x[1], reverse=True)[:11]
    return {id_to_name[pid] for pid, _ in sorted_players if pid in id_to_name}


def extract_team_features(passes: pd.DataFrame, player_pos: pd.DataFrame) -> dict:
    """
    Extract a feature vector representing a team's passing style.
    Returns a dict of metrics that can be used for clustering.
    """
 
    
    # Build network
    G = nx.DiGraph()
    edge_counts = passes.groupby(["player.name", "pass.recipient.name"]).size().reset_index(name="weight")
    for _, row in edge_counts.iterrows():
        G.add_edge(row["player.name"], row["pass.recipient.name"], weight=row["weight"])
    
    # Add pass distance
    passes = passes.copy()
    passes["pass_distance"] = np.sqrt(
        (passes["end_x"] - passes["x"])**2 + 
        (passes["end_y"] - passes["y"])**2
    )
    passes["is_forward"] = passes["end_x"] > passes["x"]
    passes["is_progressive"] = (passes["end_x"] - passes["x"]) > 10  # Progressive = >10 units forward
    
    # Calculate features
    features = {
        # Network structure
        "network_density": nx.density(G),
        "avg_clustering": nx.average_clustering(G.to_undirected()) if len(G) > 2 else 0,
        "reciprocity": nx.reciprocity(G) if len(G.edges()) > 0 else 0,
        
        # Centralization (is passing dominated by few players?)
        "degree_centralization": calculate_degree_centralization(G),
        
        # Passing directness
        "forward_pass_pct": passes["is_forward"].mean() * 100,
        "progressive_pass_pct": passes["is_progressive"].mean() * 100,
        
        # Pass distances
        "avg_pass_distance": passes["pass_distance"].mean(),
        "median_pass_distance": passes["pass_distance"].median(),
        "long_pass_pct": (passes["pass_distance"] > 30).mean() * 100,
        "short_pass_pct": (passes["pass_distance"] < 15).mean() * 100,
        
        # Spatial - average position of passes
        "avg_pass_start_x": passes["x"].mean(),
        "avg_pass_end_x": passes["end_x"].mean(),
        "territorial_gain": (passes["end_x"] - passes["x"]).mean(),
        
        # Width usage
        "pass_width_std": passes["y"].std(),
        "end_width_std": passes["end_y"].std(),
        
        # Volume
        "passes_per_possession": len(passes),  # Will normalize later
        
        # Key player dependency (Gini coefficient)
        "pass_distribution_gini": calculate_gini(
            passes.groupby("player.name").size().values
        ),
    }
    
    return features



def passes_to_heatmap_features(passes: pd.DataFrame, grid_size: int = 6) -> dict:
    """Convert passes to zone-based features."""
    if passes.empty:
        return {f"zone_{i}": 0 for i in range(grid_size * grid_size * 2)}
    
    x_bins = np.linspace(0, PITCH_X, grid_size + 1)
    y_bins = np.linspace(0, PITCH_Y, grid_size + 1)
    
    origin_hist, _, _ = np.histogram2d(passes["x"], passes["y"], bins=[x_bins, y_bins])
    origin_hist = origin_hist / (origin_hist.sum() + 1e-10)
    
    dest_hist, _, _ = np.histogram2d(passes["end_x"], passes["end_y"], bins=[x_bins, y_bins])
    dest_hist = dest_hist / (dest_hist.sum() + 1e-10)
    
    features = {}
    for i, val in enumerate(origin_hist.flatten()):
        features[f"origin_zone_{i}"] = val
    for i, val in enumerate(dest_hist.flatten()):
        features[f"dest_zone_{i}"] = val
    
    return features


def calculate_degree_centralization(G) -> float:
    """How much is passing dominated by a few players."""
    if len(G) < 2:
        return 0
    degrees = [d for _, d in G.degree(weight="weight")]
    max_degree = max(degrees)
    n = len(degrees)
    numerator = sum(max_degree - d for d in degrees)
    denominator = (n - 1) * max_degree if max_degree > 0 else 1
    return numerator / denominator if denominator > 0 else 0


def calculate_gini(values: np.ndarray) -> float:
    """Gini coefficient: 0 = equal distribution, 1 = one player dominates."""
    if len(values) == 0:
        return 0
    sorted_values = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(sorted_values)
    if cumulative[-1] == 0:
        return 0
    return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1])


# =============================================================================
# MAIN CLUSTERING PIPELINE (EFFICIENT)
# =============================================================================

def build_team_features_efficient(data: dict, approach: str = "network") -> pd.DataFrame:
    """
    Build feature matrix for all teams using cached data.
    NO additional API calls!
    
    approach: "network", "heatmap", or "combined"
    """
    teams = data["teams"]
    
    print(f"Extracting features for {len(teams)} teams (approach: {approach})...")
    
    feature_rows = []
    
    for team_name in teams:
        print(f"  Processing {team_name}...")
        
        # Get passes from cache
        passes = get_team_passes_from_cache(data, team_name)
        
        if passes.empty:
            print(f"    No passes found, skipping")
            continue
        
        # Filter to most-played XI
        top11 = get_team_top11_from_cache(data, team_name)
        passes = passes[
            passes["player.name"].isin(top11) & 
            passes["pass.recipient.name"].isin(top11)
        ]
        
        if passes.empty:
            continue
        
        # Extract features based on approach
        if approach == "network":
            features = extract_team_features(passes)
        elif approach == "heatmap":
            features = passes_to_heatmap_features(passes)
        elif approach == "combined":
            features = {**extract_team_features(passes), **passes_to_heatmap_features(passes)}
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        if features:
            features["team"] = team_name
            feature_rows.append(features)
    
    return pd.DataFrame(feature_rows)


def cluster_teams(features_df: pd.DataFrame, n_clusters: int = 4) -> tuple:
    """Cluster teams and return results."""
    feature_cols = [c for c in features_df.columns if c != "team"]
    
    X = features_df[feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df = features_df.copy()
    features_df["cluster"] = kmeans.fit_predict(X_scaled)
    
    return features_df, X_scaled, feature_cols


def visualize_team_clusters(features_df: pd.DataFrame, X_scaled: np.ndarray, 
                            feature_cols: list, title: str = "Team Style Clusters"):
    """Visualize clustering results."""
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cluster scatter
    ax = axes[0]
    colors = plt.cm.tab10(features_df["cluster"])
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=200, alpha=0.7, edgecolor="white", linewidth=2)
    
    for i, team in enumerate(features_df["team"]):
        ax.annotate(team, (X_pca[i, 0], X_pca[i, 1]), fontsize=9, ha="center", va="bottom")
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Feature importance
    ax = axes[1]
    loadings = np.sqrt(pca.components_[0]**2 + pca.components_[1]**2)
    
    # Show top 15 features
    top_idx = np.argsort(loadings)[-15:]
    top_features = [feature_cols[i] for i in top_idx]
    top_loadings = loadings[top_idx]
    
    ax.barh(range(len(top_features)), top_loadings, color="steelblue")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Features Defining Team Styles")
    
    plt.tight_layout()
    plt.show()
    
    return pca


def print_cluster_summary(features_df: pd.DataFrame):
    """Print readable cluster summary."""
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    
    for cluster_id in sorted(features_df["cluster"].unique()):
        teams = features_df[features_df["cluster"] == cluster_id]["team"].tolist()
        print(f"\n🔵 Cluster {cluster_id}:")
        print(f"   Teams: {', '.join(teams)}")
        
        # Get cluster averages for key metrics
        cluster_data = features_df[features_df["cluster"] == cluster_id]
        
        if "forward_pass_pct" in cluster_data.columns:
            print(f"   Avg Forward Pass %: {cluster_data['forward_pass_pct'].mean():.1f}%")
        if "avg_pass_distance" in cluster_data.columns:
            print(f"   Avg Pass Distance: {cluster_data['avg_pass_distance'].mean():.1f}")
        if "network_density" in cluster_data.columns:
            print(f"   Network Density: {cluster_data['network_density'].mean():.3f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main_clustering():
    """
    Main function for efficient team clustering.
    Loads data ONCE, then analyzes all teams.
    """
    # Load competitions
    comps = load_competitions()
    
    # Find Premier League 2015/16
    mask = comps["competition_name"].eq("Premier League") & \
           comps["season_name"].str.contains("2015", na=False)
    if mask.sum() == 0:
        print("Competition not found!")
        return
    
    comp_row = comps[mask].iloc[0]
    competition_id = int(comp_row["competition_id"])
    season_id = int(comp_row["season_id"])
    
    print(f"Competition: {comp_row['competition_name']} {comp_row['season_name']}")
    
    # =========================================================================
    # STEP 1: Load all data ONCE (this is the slow part)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading all match data (one-time)")
    print("=" * 60)
    
    # data = load_all_season_data(competition_id, season_id, sleep=0.1)
    data = load_all_season_data_cached(
        competition_id, season_id,
        use_async=True,
        max_workers=15
    )
    # =========================================================================
    # STEP 2: Extract features (fast - no API calls)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Extracting team features")
    print("=" * 60)
    
    # Try different approaches:
    features_df = build_team_features_efficient(data, approach="network")
    # features_df = build_team_features_efficient(data, approach="heatmap")
    # features_df = build_team_features_efficient(data, approach="combined")
    
    print(f"\nExtracted features for {len(features_df)} teams")
    print(f"Features: {len([c for c in features_df.columns if c != 'team'])}")
    
    # =========================================================================
    # STEP 3: Cluster teams (instant)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Clustering teams")
    print("=" * 60)
    
    features_df, X_scaled, feature_cols = cluster_teams(features_df, n_clusters=4)
    
    # =========================================================================
    # STEP 4: Visualize and analyze
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Visualization")
    print("=" * 60)
    
    visualize_team_clusters(features_df, X_scaled, feature_cols, 
                           title="Premier League 2015/16 - Team Playing Styles")
    print_cluster_summary(features_df)
    
    # Save results
    features_df.to_csv("team_clusters.csv", index=False)
    print("\nResults saved to team_clusters.csv")
    
    return data, features_df


if __name__ == "__main__":
    data, results = main_clustering()




# def build_all_teams_features(competition_id: int, season_id: int, 
#                               sleep: float = 0.15) -> pd.DataFrame:
#     """
#     Extract passing features for all teams in a competition.
#     """
#     matches = load_matches(competition_id, season_id)
    
#     # Get all unique teams
#     teams = set()
#     for _, m in matches.iterrows():
#         teams.add(m["home_team"]["home_team_name"])
#         teams.add(m["away_team"]["away_team_name"])
    
#     print(f"Found {len(teams)} teams")
    
#     all_features = []
    
#     for team_name in teams:
#         print(f"Processing {team_name}...")
        
#         try:
#             # Get team's matches
#             mask = matches["home_team"].apply(lambda d: d.get("home_team_name") == team_name) | \
#                    matches["away_team"].apply(lambda d: d.get("away_team_name") == team_name)
#             team_matches = matches[mask]
            
#             # Get most played XI
#             top11_names, _, _ = most_played_11_season(
#                 competition_id, season_id, team_name, sleep=0.05
#             )
            
#             # Collect all passes
#             all_passes = []
#             for _, match in team_matches.iterrows():
#                 mid = int(match["match_id"])
#                 ev = load_events(mid)
#                 p = completed_passes(ev, team_name)
#                 p = standardize_halves(p)
#                 if should_rotate_match_using_shots(ev, team_name):
#                     p = rotate_180(p)
#                 p = p[p["player.name"].isin(top11_names) & 
#                       p["pass.recipient.name"].isin(top11_names)]
#                 all_passes.append(p)
#                 time.sleep(sleep)
            
#             season_passes = pd.concat(all_passes, ignore_index=True)
            
#             # Get formation positions
#             formation_positions = get_most_common_position_season(
#                 competition_id, season_id, team_name, team_matches, sleep=0.02
#             )
            
#             # Build network and get positions
#             player_pos, _ = build_pass_network_with_formation(
#                 season_passes, formation_positions, min_edge_count=1
#             )
            
#             # Extract features
#             features = extract_team_features(season_passes, player_pos)
#             features["team"] = team_name
#             all_features.append(features)
            
#         except Exception as e:
#             print(f"  Error processing {team_name}: {e}")
#             continue
    
#     return pd.DataFrame(all_features)


# def cluster_teams(features_df: pd.DataFrame, n_clusters: int = 4):
#     """
#     Cluster teams based on their passing style features.
#     """
#     # Prepare feature matrix
#     feature_cols = [c for c in features_df.columns if c != "team"]
#     X = features_df[feature_cols].values
#     team_names = features_df["team"].values
    
#     # Handle any NaN/inf
#     X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Cluster
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(X_scaled)
    
#     # Add cluster labels to dataframe
#     features_df = features_df.copy()
#     features_df["cluster"] = clusters
    
#     return features_df, X_scaled, feature_cols


# def visualize_clusters(features_df: pd.DataFrame, X_scaled: np.ndarray, 
#                        feature_cols: list):
#     """
#     Visualize team clusters using PCA.
#     """
#     # PCA for visualization
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
    
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
#     # Scatter plot
#     ax = axes[0]
#     scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
#                          c=features_df["cluster"], cmap="tab10", s=150, alpha=0.7)
    
#     for i, team in enumerate(features_df["team"]):
#         ax.annotate(team, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, ha="center")
    
#     ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
#     ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
#     ax.set_title("Team Clusters (PCA Projection)")
    
#     # Feature importance for PCA
#     ax = axes[1]
#     loadings = pd.DataFrame(
#         pca.components_.T,
#         columns=["PC1", "PC2"],
#         index=feature_cols
#     )
#     loadings["importance"] = np.sqrt(loadings["PC1"]**2 + loadings["PC2"]**2)
#     loadings = loadings.sort_values("importance", ascending=True)
    
#     ax.barh(range(len(loadings)), loadings["importance"], color="steelblue")
#     ax.set_yticks(range(len(loadings)))
#     ax.set_yticklabels(loadings.index, fontsize=8)
#     ax.set_xlabel("Feature Importance (PCA Loading Magnitude)")
#     ax.set_title("Which Features Define Team Styles?")
    
#     plt.tight_layout()
#     plt.show()
    
#     return pca, loadings


# def plot_cluster_profiles(features_df: pd.DataFrame, feature_cols: list):
#     """
#     Show the average profile of each cluster.
#     """
#     # Normalize features for comparison
#     normalized = features_df.copy()
#     for col in feature_cols:
#         col_min = normalized[col].min()
#         col_max = normalized[col].max()
#         if col_max > col_min:
#             normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
#         else:
#             normalized[col] = 0
    
#     cluster_profiles = normalized.groupby("cluster")[feature_cols].mean()
    
#     # Radar chart for each cluster
#     n_clusters = len(cluster_profiles)
#     fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5), 
#                              subplot_kw=dict(polar=True))
    
#     if n_clusters == 1:
#         axes = [axes]
    
#     # Select most important features for radar
#     top_features = feature_cols[:8] if len(feature_cols) > 8 else feature_cols
#     angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
#     angles += angles[:1]
    
#     for idx, (cluster_id, profile) in enumerate(cluster_profiles.iterrows()):
#         ax = axes[idx]
#         values = profile[top_features].tolist()
#         values += values[:1]
        
#         ax.plot(angles, values, 'o-', linewidth=2)
#         ax.fill(angles, values, alpha=0.25)
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels([f[:12] for f in top_features], size=8)
        
#         teams_in_cluster = features_df[features_df["cluster"] == cluster_id]["team"].tolist()
#         ax.set_title(f"Cluster {cluster_id}\n({', '.join(teams_in_cluster[:3])}...)")
    
#     plt.tight_layout()
#     plt.show()


# def describe_clusters(features_df: pd.DataFrame, feature_cols: list):
#     """
#     Generate text descriptions of each cluster.
#     """
#     cluster_means = features_df.groupby("cluster")[feature_cols].mean()
#     overall_means = features_df[feature_cols].mean()
    
#     print("\n" + "="*70)
#     print("CLUSTER DESCRIPTIONS")
#     print("="*70)
    
#     style_labels = {
#         "high_forward_pass_pct": "Direct",
#         "low_forward_pass_pct": "Patient",
#         "high_short_pass_pct": "Short-passing",
#         "high_long_pass_pct": "Long-ball",
#         "high_network_density": "Fluid/Interconnected",
#         "low_network_density": "Structured",
#         "high_pass_distribution_gini": "Star-dependent",
#         "low_pass_distribution_gini": "Distributed",
#         "high_avg_pass_start_x": "High-pressing",
#         "low_avg_pass_start_x": "Deep-sitting",
#     }
    
#     for cluster_id in sorted(features_df["cluster"].unique()):
#         teams = features_df[features_df["cluster"] == cluster_id]["team"].tolist()
#         profile = cluster_means.loc[cluster_id]
        
#         print(f"\n📊 CLUSTER {cluster_id}")
#         print(f"   Teams: {', '.join(teams)}")
        
#         # Identify distinguishing characteristics
#         characteristics = []
        
#         if profile["forward_pass_pct"] > overall_means["forward_pass_pct"] + 5:
#             characteristics.append("Direct/Vertical")
#         elif profile["forward_pass_pct"] < overall_means["forward_pass_pct"] - 5:
#             characteristics.append("Patient/Possession-based")
        
#         if profile["short_pass_pct"] > overall_means["short_pass_pct"] + 5:
#             characteristics.append("Short-passing")
#         if profile["long_pass_pct"] > overall_means["long_pass_pct"] + 3:
#             characteristics.append("Long-ball tendencies")
        
#         if profile["network_density"] > overall_means["network_density"] * 1.1:
#             characteristics.append("Fluid/High connectivity")
#         elif profile["network_density"] < overall_means["network_density"] * 0.9:
#             characteristics.append("Structured/Predictable patterns")
        
#         if profile["pass_distribution_gini"] > overall_means["pass_distribution_gini"] + 0.05:
#             characteristics.append("Reliant on key players")
#         elif profile["pass_distribution_gini"] < overall_means["pass_distribution_gini"] - 0.05:
#             characteristics.append("Well-distributed involvement")
        
#         if profile["avg_pass_start_x"] > overall_means["avg_pass_start_x"] + 3:
#             characteristics.append("High defensive line")
#         elif profile["avg_pass_start_x"] < overall_means["avg_pass_start_x"] - 3:
#             characteristics.append("Deep defending")
        
#         print(f"   Style: {', '.join(characteristics) if characteristics else 'Average/Balanced'}")
        
#         # Key stats
#         print(f"   Key Stats:")
#         print(f"     • Forward Pass %: {profile['forward_pass_pct']:.1f}%")
#         print(f"     • Avg Pass Distance: {profile['avg_pass_distance']:.1f}")
#         print(f"     • Network Density: {profile['network_density']:.3f}")
#         print(f"     • Pass Distribution Gini: {profile['pass_distribution_gini']:.3f}")



# def main_clustering():
#     """Main function to cluster all teams in a competition."""
#     comps = load_competitions()
    
#     comp_row = pick_season_competition(
#         comps, preferred_name='Premier League', preferred_season="2015/2016"
#     )
#     competition_id = int(comp_row["competition_id"])
#     season_id = int(comp_row["season_id"])
    
#     print(f"Building features for all teams in {comp_row['competition_name']} {comp_row['season_name']}...")
    
#     # This will take a while (loads data for all teams)
#     features_df = build_all_teams_features(competition_id, season_id, sleep=0.1)
    
#     # Save intermediate results
#     features_df.to_csv("team_features.csv", index=False)
#     print(f"\nExtracted features for {len(features_df)} teams")
#     print(features_df.head())
    
#     # Cluster teams
#     feature_cols = [c for c in features_df.columns if c not in ["team", "cluster"]]
#     features_df, X_scaled, feature_cols = cluster_teams(features_df, n_clusters=4)
    
#     # Visualize
#     pca, loadings = visualize_clusters(features_df, X_scaled, feature_cols)
#     plot_cluster_profiles(features_df, feature_cols)
#     describe_clusters(features_df, feature_cols)
    
#     # Print cluster assignments
#     print("\n" + "="*50)
#     print("FINAL CLUSTER ASSIGNMENTS")
#     print("="*50)
#     for cluster_id in sorted(features_df["cluster"].unique()):
#         teams = features_df[features_df["cluster"] == cluster_id]["team"].tolist()
#         print(f"\nCluster {cluster_id}: {', '.join(teams)}")
    
#     return features_df


# if __name__ == "__main__":
#     # Run clustering analysis
#     features_df = main_clustering()