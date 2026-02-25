## Football Data Analysis Project

This project is a working progress analysing football data. It focuses on passing, shooting, and upsets in the English Premier League using StatsBomb open data and historical betting odds. It includes Python scripts for clustering, visualization, and statistical analysis. 

### Project Structure
- `Data/`: Contains CSV files for EPL seasons and data info.
- `Outputs/`: Stores generated plots and results.
- `src/`: Where the scripts are found
    - `Passes/`: Scripts for passing maps and team clustering.
    - `Shots/`: Scripts for shot clustering and shot map visualizations.
    - `Upsets/`: Script for analyzing upsets using match results and betting odds.

### Key Features
- Downloads and processes StatsBomb open data.
- Visualizes passing and shooting patterns using `mplsoccer`.
- Clusters teams and shots using machine learning (KMeans, PCA, UMAP, Agglomerative Clustering).
- Detects upsets based on betting odds and match outcomes.
- Uses pandas, matplotlib, seaborn, scikit-learn, and other scientific libraries.

### Getting Started
1. Install requirements:
	```bash
	pip install pandas matplotlib seaborn scikit-learn mplsoccer requests umap
	```
2. Place EPL CSV data in the `Data/` folder.
3. Run scripts in `src/` to generate analyses and plots.

### Credits
StatsBomb open data, Kaggle, and open-source Python libraries.
