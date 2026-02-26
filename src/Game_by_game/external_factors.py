import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Upsets import read_filtered_data
import plotly.express as px


IMPORTANT_COLS = [
    'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR',
    'Referee','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR'
    ]


def referee_analysis(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Analyzes the impact of referees by number of fouls, cards, penalties per match.
    '''
    # Group by referee and calculate average fouls and cards per match
    referee_stats = df.groupby('Referee').agg({
        'HF': 'mean',  # Average home fouls
        'AF': 'mean',  # Average away fouls
        'HY': 'mean',  # Average home yellow cards
        'AY': 'mean',  # Average away yellow cards
        'HR': 'mean',  # Average home red cards
        'AR': 'mean',   # Average away red cards
        'FTR': lambda x: (x == 'H').mean(),  # Proportion of home wins
    }).reset_index()
    # Count total matches officiated by each referee

    match_counts = df['Referee'].value_counts().reset_index()
    match_counts.columns = ['Referee', 'Matches_Officiated']

    # Merge match counts with referee stats
    referee_stats = referee_stats.merge(match_counts, left_on='Referee', right_on='Referee', how='left')

    # Calculate total fouls and cards per match
    referee_stats['Total_Fouls'] = referee_stats['HF'] + referee_stats['AF']
    referee_stats['Total_Cards'] = referee_stats['HY'] + referee_stats['AY'] + referee_stats['HR'] + referee_stats['AR']
    
    # Sort referees by total cards
    sorted_referees = referee_stats.sort_values(by='Total_Cards', ascending=False)
    
    return sorted_referees[['Referee','Total_Fouls','Total_Cards','Matches_Officiated', 'FTR']]

def main():
    season_start, season_end = 20,25
    seasons = [f'{y}_{y+1}' for y in range(season_start, season_end)]

    # Initialise a df for storing referee stats across seasons
    all_referee_stats = pd.DataFrame()

    # Process each season and concatenate referee stats    
    for season in seasons:
        print('Processing season:', season)
        file_path = f'Data/PL_{season}.csv'
        df = read_filtered_data(file_path)
        ref_df = referee_analysis(df)
        ref_df['Season'] = season  # Add season column
        all_referee_stats = pd.concat([all_referee_stats, ref_df[['Referee', 'Total_Fouls', 'Total_Cards', 'Matches_Officiated', 'Season', 'FTR']]], ignore_index=True)
    
    # Ensure refs with the same name across seasons are treated as the same ref
    all_referee_stats = all_referee_stats.groupby('Referee').agg({
        'Total_Fouls': 'mean', # Average fouls across seasons
        'Total_Cards': 'mean',  # Average cards across seasons
        'Matches_Officiated': 'sum', # Total matches officiated across seasons
        'Season': lambda x: list(x), # List of seasons officiated
        'FTR': 'mean'  # Average home win proportion across seasons (already numeric)
    }).reset_index()

    # Remove refs who officiated less than 10 matches across all seasons to focus on more experienced refs
    all_referee_stats = all_referee_stats[all_referee_stats['Matches_Officiated'] >= 10]
    
    # Print top referees across all seasons
    # print("Top referees across all seasons based on total fouls:")
    # print(all_referee_stats.sort_values(by='Total_Fouls', ascending=False).head(20))

    # Create visualisations of referee stats
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='Referee', y='Total_Fouls', data=all_referee_stats.sort_values(by='Total_Fouls', ascending=False).head(20))
    # # colour each bar based on the number of matches officiated (darker colour for more matches)
    # colors = plt.cm.coolwarm(np.linspace(0, 1, 20))
    # bars = plt.gca().patches
    # for i, bar in enumerate(bars):
    #     bar.set_color(colors[i])

    # plt.xticks(rotation=45, ha='right')
    # plt.title('Top 20 Referees by Average Fouls per Match')
    # plt.xlabel('Referee')
    # plt.ylabel('Average Fouls per Match')
    # plt.tight_layout()
    # # plt.savefig('Outputs/Top_Referees_Fouls.png')
    # plt.show()

    # Plot scatter of total fouls vs home win proportion, colored by number of matches officiated
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        all_referee_stats['Total_Fouls'],
        all_referee_stats['FTR'], 
        s=all_referee_stats['Matches_Officiated']*10, 
        alpha=0.6,
        )
    
    # annotate each point with the referee name
    # for i, row in all_referee_stats.iterrows():
        # plt.annotate(row['Referee'], (row['Total_Fouls'], row['FTR']), fontsize=8, alpha=0.7)
    # avoid overlapping labels by adjusting the position of the annotations
    for i, row in all_referee_stats.iterrows():
        offset = np.random.choice([-0.01, 0.01])
        plt.annotate(row['Referee'], (row['Total_Fouls'], row['FTR']+offset), fontsize=8, alpha=0.7)
   
    plt.xlabel('Average Total Fouls per Match')
    plt.ylabel('Average Home Win Proportion')
    plt.title('Referee Fouls vs Home Win Proportion (Size = Matches Officiated)')
    plt.tight_layout()
    plt.savefig('Outputs/Referee_Fouls_vs_Home_Win_Proportion.png')
    plt.show()

    # Use seaborn to create an interactive scatter plot of total fouls vs home win proportion, colored by number of matches officiated with labels for each name
    # plt.figure(figsize=(10, 6))
    # scatter = sns.scatterplot(
    #     data=all_referee_stats,
    #     x='Total_Fouls',
    #     y='FTR',
    #     size='Matches_Officiated',
    #     hue='Matches_Officiated',
    #     alpha=0.6,
    #     palette='viridis'
    # )
    # # Add labels for each referee name
    # for i, row in all_referee_stats.iterrows():
    #     plt.annotate(row['Referee'], (row['Total_Fouls'], row['FTR']), fontsize=8, alpha=0.7)

    # plt.xlabel('Average Total Fouls per Match')
    # plt.ylabel('Average Home Win Proportion')
    # plt.title('Referee Fouls vs Home Win Proportion (Size & Color = Matches Officiated)')
    # plt.legend(title='Matches Officiated', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig('Outputs/Referee_Fouls_vs_Home_Win_Proportion_Seaborn.png')
    # plt.show()


if __name__ == "__main__":
    main()