import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mplsoccer import Pitch
from collections import defaultdict


BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
PITCH_X, PITCH_Y = 120, 80

# Add these constants at the top of your file

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
    """Rotate coordinates 180 degrees (x -> 120-x, y -> 80-y). This is essential for passes occuring in the second half."""
    df = df.copy()
    df["x"] = PITCH_X - df["x"]
    df["y"] = PITCH_Y - df["y"]
    df["end_x"] = PITCH_X - df["end_x"]
    df["end_y"] = PITCH_Y - df["end_y"]
    return df


def resolve_position_conflicts(formation_positions: dict, min_distance: float = 12.0) -> dict:
    """
    If multiple players end up at the same/similar position, spread them out.
    """
    if not formation_positions:
        return formation_positions
    
    resolved = dict(formation_positions)
    players = list(resolved.keys())
    
    # Multiple passes to resolve conflicts
    for _ in range(20):
        changed = False
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                x1, y1 = resolved[p1]
                x2, y2 = resolved[p2]
                
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if dist < min_distance:
                    # Push apart perpendicular to the goal line (mainly in y)
                    if dist == 0:
                        # Identical positions - offset in y
                        resolved[p1] = (x1, y1 - min_distance/2)
                        resolved[p2] = (x2, y2 + min_distance/2)
                    else:
                        dx = (x2 - x1) / dist
                        dy = (y2 - y1) / dist
                        push = (min_distance - dist) / 2 + 1
                        
                        resolved[p1] = (x1 - dx * push, y1 - dy * push)
                        resolved[p2] = (x2 + dx * push, y2 + dy * push)
                    
                    changed = True
        
        if not changed:
            break
    
    # Clamp to pitch
    for player in resolved:
        x, y = resolved[player]
        resolved[player] = (
            np.clip(x, 5, PITCH_X - 5),
            np.clip(y, 5, PITCH_Y - 5)
        )
    
    return resolved



def completed_passes(events: pd.DataFrame, team_name: str, period: int = None) -> pd.DataFrame:
    ev = events.copy()
    ev = add_xy(ev)

    passes = ev[(ev["type.name"] == "Pass") & (ev["team.name"] == team_name)].copy()

    if period is not None:
        passes = passes[passes["period"] == period]
    # Completed passes have no outcome (no "Incomplete" outcome)
    if "pass.outcome.name" in passes.columns:
        passes = passes[passes["pass.outcome.name"].isna()]

    # Drop passes with missing essential data
    passes = passes.dropna(subset=["player.name", "pass.recipient.name", "x", "y", "end_x", "end_y"])
    
    # Remove restarts/set pieces for a cleaner “open play” network
    set_piece_types = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
    if "pass.type.name" in passes.columns:
        passes = passes[~passes["pass.type.name"].isin(set_piece_types)]

    return passes

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
    """
    Returns True if (after half normalization) the team's shots are mostly in the LEFT half,
    meaning we should rotate the whole match to make them attack RIGHT.
    """
    ev = events.copy()
    ev = add_xy(ev)  # gives x,y for all events; end_x/end_y only for passes

    shots = ev[(ev["type.name"] == "Shot") & (ev["team.name"] == team_name)].copy()
    if shots.empty:
        # fallback: use pass end_x as a weaker signal if no shots
        passes = ev[(ev["type.name"] == "Pass") & (ev["team.name"] == team_name)].copy()
        passes = passes.dropna(subset=["end_x"])
        if passes.empty:
            return False
        passes = standardize_halves(passes)
        return passes["end_x"].median() < (PITCH_X / 2)

    shots = shots.dropna(subset=["x", "y"])
    shots = standardize_halves(shots)
    return shots["x"].median() < (PITCH_X / 2)

def standardize_direction_full_match(passes: pd.DataFrame) -> pd.DataFrame:
    p = passes.copy()
    # flip 2nd half; and flip 2nd period of extra time if it exists
    flip_periods = {2, 4}
    to_flip = p["period"].isin(flip_periods)
    p.loc[to_flip, ["x","y","end_x","end_y"]] = p.loc[to_flip].assign(
        x=lambda d: PITCH_X - d["x"],
        y=lambda d: PITCH_Y - d["y"],
        end_x=lambda d: PITCH_X - d["end_x"],
        end_y=lambda d: PITCH_Y - d["end_y"],
    )[["x","y","end_x","end_y"]].values
    return p


def build_pass_network_with_formation(passes: pd.DataFrame, 
                                       formation_positions: dict,
                                       min_edge_count: int = 0,
                                       blend_weight: float = 0.3) -> tuple:
    """
    Build pass network using formation positions blended with actual pass data.
    
    blend_weight: 0.0 = pure formation positions, 1.0 = pure pass data positions
    """
    
    # Resolve any formation position conflicts
    formation_positions = resolve_position_conflicts(formation_positions, min_distance=12.0)

    # Single out the GK and fix the position to formation position
    goalkeeper_name = None
    if formation_positions:
        gk_candidates = [(name, pos) for name, pos in formation_positions.items() if pos[0] <= 15]
        if gk_candidates:
            # Pick the one closest to the goal
            goalkeeper_name = min(gk_candidates, key=lambda x: x[1][0])[0]



    # --- Calculate data-based positions (from passes) ---
    touch_from = (
        passes[["player.name", "x", "y"]]
        .rename(columns={"player.name": "player"})
    )
    touch_to = (
        passes[["pass.recipient.name", "end_x", "end_y"]]
        .rename(columns={"pass.recipient.name": "player", "end_x": "x", "end_y": "y"})
    )
    touches = pd.concat([touch_from, touch_to], ignore_index=True).dropna(subset=["x", "y"])
    
    data_positions = touches.groupby("player").agg(
        x_data=("x", "median"),
        y_data=("y", "median"),
    ).reset_index()
    
    # --- Calculate involvement stats ---
    passes_made = passes.groupby("player.name").size().reset_index(name="passes")
    passes_made = passes_made.rename(columns={"player.name": "player"})
    
    passes_received = passes.groupby("pass.recipient.name").size().reset_index(name="receptions")
    passes_received = passes_received.rename(columns={"pass.recipient.name": "player"})
    
    player_pos = (
        data_positions
        .merge(passes_made, on="player", how="left")
        .merge(passes_received, on="player", how="left")
        .fillna({"passes": 0, "receptions": 0})
    )
    player_pos["involvement"] = player_pos["passes"] + player_pos["receptions"]
    
    # --- Blend with formation positions ---
    def get_blended_position(row):
        player = row["player"]
        x_data = row["x_data"]
        y_data = row["y_data"]
        
        if player in formation_positions:
            x_form, y_form = formation_positions[player]
            
            # GOALKEEPER: Always use pure formation position (no blending)
            if player == goalkeeper_name:
                x_final = x_form
                y_final = y_form
            else:
                x_final = blend_weight * x_data + (1 - blend_weight) * x_form
                y_final = blend_weight * y_data + (1 - blend_weight) * y_form
        else:
            # No formation data, use pure data position
            x_final = x_data
            y_final = y_data
        
        return pd.Series({"x_mean": x_final, "y_mean": y_final})
    
    blended = player_pos.apply(get_blended_position, axis=1)
    player_pos["x_mean"] = blended["x_mean"]
    player_pos["y_mean"] = blended["y_mean"]
    
    # --- Final pass: resolve any remaining overlaps after blending ---
    final_positions = {row["player"]: (row["x_mean"], row["y_mean"]) 
                       for _, row in player_pos.iterrows()}
    
    # Store GK position before conflict resolution
    gk_pos_backup = final_positions.get(goalkeeper_name) if goalkeeper_name else None
    
    final_positions = resolve_position_conflicts(final_positions, min_distance=10.0)
    
    # Restore GK position if it was moved
    if goalkeeper_name and gk_pos_backup:
        final_positions[goalkeeper_name] = gk_pos_backup

    for idx, row in player_pos.iterrows():
        if row["player"] in final_positions:
            player_pos.loc[idx, "x_mean"] = final_positions[row["player"]][0]
            player_pos.loc[idx, "y_mean"] = final_positions[row["player"]][1]
   
    # --- Build edges ---
    edges = (
        passes.groupby(["player.name", "pass.recipient.name"])
        .size().reset_index(name="count")
        .rename(columns={"player.name": "passer", "pass.recipient.name": "receiver"})
    )
    edges = edges[edges["count"] >= min_edge_count]
    
    # Merge positions onto edges
    edges = edges.merge(
        player_pos[["player", "x_mean", "y_mean"]],
        left_on="passer", right_on="player", how="left"
    ).rename(columns={"x_mean": "x1", "y_mean": "y1"}).drop(columns=["player"])
    
    edges = edges.merge(
        player_pos[["player", "x_mean", "y_mean"]],
        left_on="receiver", right_on="player", how="left"
    ).rename(columns={"x_mean": "x2", "y_mean": "y2"}).drop(columns=["player"])
    
    return player_pos, edges

def plot_pass_network(player_pos: pd.DataFrame, edges: pd.DataFrame, title: str):
    pitch = Pitch(pitch_type="statsbomb", line_color="#222222")
    fig, ax = pitch.draw(figsize=(10, 7))

    # Edge widths scaled by pass counts
    if len(edges):
        lw = 0.5 + (edges["count"] / edges["count"].max()) * 6
        for i, row in edges.reset_index(drop=True).iterrows():
            pitch.lines(row["x1"], row["y1"], row["x2"], row["y2"],
                        lw=lw.iloc[i], color="#1f77b4", alpha=0.5, ax=ax, zorder=1)

    # Node sizes scaled by involvement
    sizes = 200 + (player_pos["involvement"] / player_pos["involvement"].max()) * 1200
    pitch.scatter(player_pos["x_mean"], player_pos["y_mean"],
                  s=sizes, color="#a65c5c", edgecolor="white", linewidth=1.5,
                  ax=ax, zorder=2)

    # Labels
    for _, r in player_pos.iterrows():
        ax.text(r["x_mean"], r["y_mean"], r["player"].split()[-1],
                ha="center", va="center", color="Black", fontsize=9, zorder=3)

    ax.set_title(title)
    plt.show()



def single_match_pass_network_v2(match: pd.Series, team_name: str):
    """Single match pass network using formation positions."""
    home_team = match["home_team"]["home_team_name"]
    away_team = match["away_team"]["away_team_name"]
    match_id = int(match["match_id"])
    
    events = load_events(match_id)
    top11_names, mins_table = most_played_11_match(events, team_name)
    
    # Get formation positions from this match
    formation_positions = get_player_positions_from_lineup(events, team_name)
    
    # Get passes
    passes = completed_passes(events, team_name)
    passes = standardize_halves(passes)
    if should_rotate_match_using_shots(events, team_name):
        passes = rotate_180(passes)
    
    # Filter to top 11
    passes = passes[
        passes["player.name"].isin(top11_names) &
        passes["pass.recipient.name"].isin(top11_names)
    ]
    
    # Build network with formation positions
    # blend_weight: 0.0 = pure formation, 0.3 = mostly formation with some data influence
    nodes, edges = build_pass_network_with_formation(
        passes, 
        formation_positions,
        min_edge_count=3,
        blend_weight=0.3  # Adjust this: lower = more formation-based
    )
    
    plot_pass_network(
        nodes, edges,
        title=f"Pass network (formation-based) — {team_name}\nmatch_id={match_id} ({home_team} vs {away_team})"
    )


def full_season_pass_network_v2(competition_id: int, season_id: int, 
                                 team_name: str, sleep: float = 0.2):
    """Full season pass network using most common formation positions."""
    
    top11_names, mins_season, team_matches = most_played_11_season(
        competition_id, season_id, team_name, sleep=sleep
    )
    
    print(f"Getting formation positions for {team_name}...")
    # This will take a moment as it loads all matches again
    # In production, you'd cache this
    formation_positions = get_most_common_position_season(
        competition_id, season_id, team_name, team_matches, sleep=0.05
    )
    
    print(f"Found positions for {len(formation_positions)} players")
    
    all_passes = []
    for _, match in team_matches.iterrows():
        mid = int(match["match_id"])
        ev = load_events(mid)
        p = completed_passes(ev, team_name)
        
        p = standardize_halves(p)
        if should_rotate_match_using_shots(ev, team_name):
            p = rotate_180(p)
        
        p = p[p["player.name"].isin(top11_names) & p["pass.recipient.name"].isin(top11_names)]
        all_passes.append(p)
        time.sleep(sleep)
    
    season_passes = pd.concat(all_passes, ignore_index=True)
    
    nodes, edges = build_pass_network_with_formation(
        season_passes,
        formation_positions,
        min_edge_count=30,
        blend_weight=0.4  # Slightly more data influence for season aggregate
    )
    
    plot_pass_network(nodes, edges, f"Season pass network (formation-based) — {team_name}")


def main_4():
    """Main function using formation-based positioning."""
    comps = load_competitions()
    
    comp_row = pick_season_competition(comps, preferred_name='Premier League', preferred_season="2015/2016")
    competition_id = int(comp_row["competition_id"])
    season_id = int(comp_row["season_id"])
    
    print("Using:", comp_row[["competition_name", "season_name", "country_name"]].to_dict())
    
    matches = load_matches(competition_id, season_id)
    
    # Choose a random match
    rand_idx = np.random.choice(len(matches), size=1, replace=False)
    match = matches.iloc[rand_idx[0]]
    
    team_name = match['home_team']['home_team_name']
    print(f"Team: {team_name}")
    
    # Single match with formation positions
    single_match_pass_network_v2(match, team_name)
    
    # Full season with formation positions
    # full_season_pass_network_v2(competition_id, season_id, team_name)


if __name__ == "__main__":
    main_4()