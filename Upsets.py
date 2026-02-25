import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub


IMPORTANT_COLS = [
    'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR',
    'Referee','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR', 'B365H',
    'B365D', 'B365A'
    ]

def read_filtered_data(file_path: str = 'Data/PL_24_25.csv') -> pd.DataFrame:
    '''    
    Reads the CSV file and removes all other odds except Bet365 odds
    '''
    df = pd.read_csv(file_path)

    # Filter out unimportant columns
    df = df[IMPORTANT_COLS]
    
    # Remove rows with missing values
    df = df.dropna()
    
    return df

def upset_filtering(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Filters the dataframe to only include the columns relevant for upsets (using Bet365 odds) 
    '''
    possible_upsets = df[['FTR', 'B365H', 'B365D', 'B365A']].copy()

    # Define upsets mask based on Bet365 odds
    upsets_mask = (
        ((possible_upsets['FTR'] == 'H') & (possible_upsets['B365H'] > possible_upsets[['B365D', 'B365A']].min(axis=1))) |
        ((possible_upsets['FTR'] == 'D') & (possible_upsets['B365D'] > possible_upsets[['B365H', 'B365A']].min(axis=1))) |
        ((possible_upsets['FTR'] == 'A') & (possible_upsets['B365A'] > possible_upsets[['B365H', 'B365D']].min(axis=1)))
    )

    return possible_upsets[upsets_mask], upsets_mask

def find_shared_columns(seasons: list) -> set:
    '''
    Finds the shared columns across multiple seasons
    '''
    
    columns_summary = {}
    for season in seasons:
        file_path = f'Data/PL_{season}.csv'
        df = pd.read_csv(file_path)
        columns_summary[season] = df.columns.tolist()

    # Find which columns are common across all seasons
    common_columns = set(columns_summary[seasons[0]])
    for season in seasons[1:]:
        common_columns.intersection_update(columns_summary[season])
    
    print("Common columns across all seasons:")
    print(common_columns)
    
    # Convert to list before returning
    return list(common_columns)


def upsets_main(): 
    seasons = ['24_25', '23_24', '22_23', '21_22', '20_21'
               ,'19_20', '18_19']
    columns_summary = {}
    
    num_upsets = np.zeros(len(seasons))
    for i,season in enumerate(seasons):
        file_path = f'Data/PL_{season}.csv'
        df = read_filtered_data(file_path)
        df, upset_mask = upset_filtering(df)
        columns_summary[season] = upset_mask.index.tolist()
        num_upsets[i] = upset_mask.sum()
        print(f"Season {season} has {num_upsets[i]} upsets.")

    plt.figure()
    plt.bar(seasons, num_upsets, color='skyblue')
    plt.xlabel('Season')
    plt.ylabel('Number of Upsets')
    plt.title('Number of Upsets per Season in Premier League')
    plt.show()

def kaggle_main():
    # Example usage of kagglehub to download dataset
    path = kagglehub.dataset_download("secareanualin/football-events")
    # print("Dataset downloaded to:", path)

    # Read data from the dataset
    df = pd.read_csv(f"{path}/events.csv")

    # Read data dictionary
    with open(path + "/dictionary.txt", 'r') as f:
        data_key = f.readlines()
    print(data_key)


if __name__ == "__main__":
    # upsets_main()
    kaggle_main()
    


