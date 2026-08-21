import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
import importlib
import Data_processing.statsbomb_to_preferred_names as preferred_names
import os



def get_player_nickname(main_events_df, teamname, player_name=None, redo=False):
    events_df = main_events_df[main_events_df["team_name"] == teamname].copy()
    # Check if the CSV exists, and create it if not
    output_dir = rf"Data_processing\name_mappings\{teamname.lower()}_mapping.csv"

    if not os.path.exists(output_dir) or redo:
        # Match players using wikidata and fuzzy matching, saving as a CSV file
        print(f"Creating new mapping for {teamname} and saving to {output_dir}")
        matched = preferred_names.create_player_name_mapping(
            events_df,
            teamname,
            output_path=output_dir,
        )
    else:
        # Load the existing CSV file
        print(f"Loading existing mapping from {output_dir}")
        matched = pd.read_csv(output_dir)

    # Return the specific player name string if provided, otherwise return statsbomb_name:wikidata_preferred_name dictionary
    if player_name:
        player_row = matched[matched['statsbomb_name'] == player_name]
        if not player_row.empty:
            return player_row.iloc[0]['wikidata_preferred_name']
        else:
            print(f"No mapping found for {player_name}")
            return None

    else:
        return dict(zip(matched['statsbomb_name'], matched['wikidata_preferred_name']))
        
def draw_pitch(ax, draw_half=False):
    """Draw pitch lines """
    pitch = Pitch(half=draw_half, pitch_color='white', line_color='black', line_alpha=0.6)
    pitch.draw(ax)


def get_short_name(name: str) -> str:
    parts = name.split()

    # Alter chosen name of specific players using external file

    # Remove common prefixes from the last name if present
    if len(parts) >= 2 and parts[-2].lower() in [
        "de", "da", "del", "van", "von", "dos", "di", "la", "le", "ter"
    ]:
        return " ".join(parts[-2:])
    
    
    return parts[-1]
