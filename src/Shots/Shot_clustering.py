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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import umap


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
    # Select relevant features
    features = ['x', 'y', 'shot.statsbomb_xg', 'shot.body_part.name', 'shot.technique.name', 'period']
    feature_matrix = shots_df[features].copy()

    # Handle variables
    feature_matrix = pd.get_dummies(feature_matrix, columns=['shot.body_part.name', 'shot.technique.name', 'period'], drop_first=True)
    feature_matrix.fillna(0, inplace=True)
    feature_matrix = feature_matrix.astype(float)
    feature_matrix = np.array(feature_matrix.values)

    return feature_matrix


def build_team_feature_matrix(shots_df):

    goal_x, goal_y = 120, 40
    shots_df["distance"] = np.sqrt(
        (goal_x - shots_df["x"])**2 +
        (goal_y - shots_df["y"])**2
    )

    teams = shots_df["team.name"].unique()

    team_features = []
    team_names = []

    for team in teams:
        df = shots_df[shots_df["team.name"] == team]

        matches = df["match_id"].nunique()
        total_shots = len(df)
        total_xg = df["shot.statsbomb_xg"].sum()
        goals = (df["shot.outcome.name"] == "Goal").sum()

        # Volume
        shots_per_match = total_shots / matches
        xg_per_match = total_xg / matches
        xg_per_shot = total_xg / total_shots if total_shots else 0
        conversion = goals / total_shots if total_shots else 0

        # Distance
        avg_dist = df["distance"].mean()
        std_dist = df["distance"].std()

        # Location
        inside_box = (df["x"] >= 102) & (df["y"].between(18, 62))
        pct_inside_box = inside_box.mean()

        # Context
        pct_under_pressure = df["under_pressure"].mean()
        pct_first_time = df["shot.first_time"].mean()
        pct_dribble = df["shot.follows_dribble"].mean()

        # Play pattern
        open_play = (df["shot.type.name"] == "Open Play").mean()
        corner = (df["shot.type.name"] == "Corner").mean()
        free_kick = (df["shot.type.name"] == "Free Kick").mean()

        # Big chances
        pct_big_chance = (df["shot.statsbomb_xg"] > 0.3).mean()

        # Headers
        pct_headers = (df["shot.body_part.name"] == "Head").mean()

        features = [
            shots_per_match,
            xg_per_match,
            xg_per_shot,
            conversion,
            avg_dist,
            std_dist,
            pct_inside_box,
            pct_under_pressure,
            pct_first_time,
            pct_dribble,
            open_play,
            corner,
            free_kick,
            pct_big_chance,
            pct_headers
        ]
        team_features.append(features)
        team_names.append(team)

    return np.array(team_features), team_names


def build_team_feature_matrix1(shots_df):
    teams = shots_df['team.name'].unique()
    team_features = []
    team_names = []
    for team in teams:
        team_df = shots_df[shots_df['team.name'] == team]
        # Features: mean shot x/y, mean xG, body part proportions, technique proportions
        mean_x = team_df['x'].mean()
        mean_y = team_df['y'].mean()
        std_x = team_df['x'].std()
        std_y = team_df['y'].std()
        mean_xg = team_df['shot.statsbomb_xg'].mean()
        std_xg = team_df['shot.statsbomb_xg'].std()
        # Outside box percentage
        # _, pct_outside = outside_box_shots(team_df)
        # Body part proportions
        body_parts = team_df['shot.body_part.name'].value_counts(normalize=True)
        foot_pct = body_parts.get('Foot', 0)
        head_pct = body_parts.get('Head', 0)
        other_pct = body_parts.get('Other', 0)
        # Technique proportions
        techniques = team_df['shot.technique.name'].value_counts(normalize=True)
        normal_pct = techniques.get('Normal', 0)
        volley_pct = techniques.get('Volley', 0)
        chip_pct = techniques.get('Chip', 0)
        # Assemble feature vector
        features = [mean_x, mean_y, mean_xg, foot_pct, head_pct, other_pct, normal_pct, volley_pct, chip_pct, std_x, std_y, std_xg]
        team_features.append(features)
        team_names.append(team)
    feature_matrix = np.array(team_features)
    return feature_matrix, team_names


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
    matches, chosen_season = get_matches_season(chosen_season=['La Liga','2015/2016'])

    # Get shots for the season
    shots_df = get_season_shots(matches, season=chosen_season)
    print(shots_df.columns)

    # Build team-level feature matrix
    X_team, team_names = build_team_feature_matrix(shots_df)

    # Deal with any Nan values
    imputer = SimpleImputer(strategy="median")
    X_team = imputer.fit_transform(X_team)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_team)

    # Debug check for Nan
    # df_debug = pd.DataFrame(X_scaled)
    # print(df_debug[df_debug.isna().any(axis=1)])

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f'Explained variance ratio: {pca.explained_variance_ratio_}')

    # Apply UMAP for visualization
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)


    # Try different numbers of clusters and evaluate with silhouette score
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        k_means_score = silhouette_score(X_scaled, labels)
        print(f'K-Means with {k} clusters: {k_means_score}')
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled)
        print(f'AgglomerativeClustering with {k} clusters: {silhouette_score(X_scaled, labels)}')



    # Apply clustering
    kmeans_clusters = 3
    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    agg_clusters = 2
    agglomerative = AgglomerativeClustering(n_clusters=agg_clusters)
    cluster_labels_agg = agglomerative.fit_predict(X_scaled)
    

    # Plot teams in PCA space colored by cluster
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set1', s=100)
    for i, team in enumerate(team_names):
        plt.text(X_pca[i, 0], X_pca[i, 1], team, fontsize=9)
    plt.title('Team Clustering by Shooting Characteristics')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(f'Outputs/Kmeans_shot_clustering_{chosen_season[0]}.png')
    plt.show()

    # Plot teams in PCA space colored by cluster
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels_agg, palette='Set1', s=100)
    for i, team in enumerate(team_names):
        plt.text(X_pca[i, 0], X_pca[i, 1], team, fontsize=9)
    plt.title('Team Clustering by Shooting Characteristics')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(f'Outputs/Agg_shot_clustering_{chosen_season[0]}.png')
    plt.show()

    # # Plot teams in UMAP space colored by cluster
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=cluster_labels, palette='Set1', s=100)
    # for i, team in enumerate(team_names):
    #     plt.text(X_umap[i, 0], X_umap[i, 1], team, fontsize=9)
    # plt.title('Team Clustering by Shooting Characteristics (UMAP)')
    # plt.xlabel('UMAP Component 1')
    # plt.ylabel('UMAP Component 2')
    # plt.legend(title='Cluster')
    # plt.tight_layout()
    # plt.savefig(f'Outputs/UMAP_shot_clustering_{chosen_season[0]}.png')
    # plt.show()
    

    # Print teams in each cluster for KMeans
    for c in range(kmeans_clusters):
        print(f"\nKMeans Cluster {c}:")
        for i, team in enumerate(team_names):
            if cluster_labels[i] == c:
                print(f"  {team}")

    # Print teams in each cluster for Agglomerative Clustering
    for c in range(agg_clusters):
        print(f"\nAgglomerative Cluster {c}:")
        for i, team in enumerate(team_names):
            if cluster_labels_agg[i] == c:
                print(f"  {team}")

if __name__ == "__main__":
    # comparison_main()
    cluster_main()