import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mplsoccer import Pitch
from collections import defaultdict
import os
from Shot_maps import get_matches_season, get_season_shot_map, plot_shot_map, get_season_shots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


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


def build_feature_matrix(shots_df):
    # Selecti relevant features
    features = ['x', 'y', 'shot.statsbomb_xg', 'shot.body_part.name', 'shot.technique.name', 'period']
    feature_matrix = shots_df[features].copy()
    
    # Handle categorical variables
    feature_matrix = pd.get_dummies(feature_matrix, columns=['shot.body_part.name', 'shot.technique.name', 'period'], drop_first=True)
    
    # Handle any missing values
    feature_matrix.fillna(0, inplace=True)
    
    # Ensure all features are numeric
    feature_matrix = feature_matrix.astype(float)  

    # Convert to numpy array for clustering 
    feature_matrix = np.array(feature_matrix.values)

    return feature_matrix


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
    
def comparison_main():
    # Get matches for season
    matches, chosen_season = get_matches_season(chosen_season=['Premier League','2015/2016'])
    team_names = ['Swansea City', 'Liverpool', 'Manchester City']
    
    map_comparison(matches, season=chosen_season, team_names=team_names)
    # g, ng, xg = get_season_shot_map(matches, season=chosen_season, team_name='Swansea City')
    
def cluster_main():
    # Get matches for the season
    matches, chosen_season = get_matches_season(chosen_season=['Premier League','2015/2016'])

    # Get shots for the season
    shots_df = get_season_shots(matches, season=chosen_season)
    print(shots_df.columns)

    # Create feature matrix for clustering
    X = build_feature_matrix(shots_df)
    labels = shots_df['team.name'].values

    # Split into training and test sets
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # plot PCA results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=labels[:len(X_train)], palette='Set1', alpha=0.7)
    plt.title('PCA of Shot Features (Training Set)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


    # # Apply KMeans clustering
    # kmeans = KMeans(n_clusters=5, random_state=42)
    # kmeans.fit(X_train)
    
    # # Predict cluster labels for test set
    # test_labels = kmeans.predict(X_test)

    
if __name__ == "__main__":
    # comparison_main()
    cluster_main()