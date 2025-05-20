import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA # For visualization
from scipy import stats # For Z-score
import os
import json

# --- 0. Configuration ---
BASE_OUTPUT_DIR = "data/turnout_analysis_results" # From script 7
INPUT_DATA_FILE = os.path.join(BASE_OUTPUT_DIR, "data_with_turnout.csv")

OUTLIER_CLUSTER_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "outlier_cluster_analysis")
os.makedirs(OUTLIER_CLUSTER_OUTPUT_DIR, exist_ok=True)

# Z-score thresholds
Z_SCORE_THRESHOLD = 3.0
MODIFIED_Z_SCORE_THRESHOLD = 3.5 # Common threshold for modified Z-score

# Isolation Forest parameters
ISO_FOREST_CONTAMINATION = 0.03 # Or a float e.g., 0.01 for 1% outliers, 0.05 for 5%
ISO_FOREST_RANDOM_STATE = 42

# DBSCAN parameters (these often require tuning based on the dataset)
DBSCAN_EPS = 0.5       # Max distance between samples for one to be considered as in the neighborhood of the other.
DBSCAN_MIN_SAMPLES = 10 # Number of samples in a neighborhood for a point to be considered as a core point.

# Features to use for multivariate outlier detection (Isolation Forest, DBSCAN)
# Choose numerical features that might collectively indicate anomalies.
# 'Turnout_Percentage' is definitely key.
# Other examples: 'liczba_uprawnionych', 'liczba_kart_wydanych_lacznie', 'liczba_glosow_waznych_na_kandydatow'
# 'Frekwencja' might be redundant if 'Turnout_Percentage' is already there.
# We will scale these features.
MULTIVARIATE_FEATURES = [
    'Turnout_Percentage',
    'liczba_uprawnionych',
    'liczba_kart_wydanych_lacznie',
    'liczba_glosow_waznych_na_kandydatow'
    # Add other relevant numeric features if available and appropriate
]

# --- 1. Load Data ---
print("--- Loading Data ---")
try:
    df = pd.read_csv(INPUT_DATA_FILE)
    print(f"Successfully loaded data: {INPUT_DATA_FILE}")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: File not found: {INPUT_DATA_FILE}. Please run script 7 first.")
    exit()
except Exception as e:
    print(f"Error loading {INPUT_DATA_FILE}: {e}")
    exit()

# --- 2. Data Preparation for Models ---
# Ensure selected features exist and handle missing values
df_analysis = df.copy()
missing_mv_features = [f for f in MULTIVARIATE_FEATURES if f not in df_analysis.columns]
if missing_mv_features:
    print(f"ERROR: The following features for multivariate analysis are missing: {missing_mv_features}")
    # Fallback or exit
    if 'Turnout_Percentage' in df_analysis.columns:
        print("Warning: Proceeding with multivariate analysis using only 'Turnout_Percentage'.")
        current_mv_features = ['Turnout_Percentage']
    else:
        print("Cannot proceed without 'Turnout_Percentage' for multivariate analysis.")
        exit()
else:
    current_mv_features = MULTIVARIATE_FEATURES

# For simplicity, drop rows with NaNs in the features used for model-based detection
# A more robust approach might involve imputation.
df_analysis.dropna(subset=current_mv_features, inplace=True)
print(f"Shape after dropping NaNs in features {current_mv_features}: {df_analysis.shape}")

if df_analysis.empty:
    print("No data left after removing NaNs. Cannot proceed.")
    exit()

# Scale features for Isolation Forest and DBSCAN
scaler = StandardScaler()
df_analysis_scaled = df_analysis.copy() # Keep original values in df_analysis for Z-score on Turnout_Percentage
df_analysis_scaled[current_mv_features] = scaler.fit_transform(df_analysis[current_mv_features])


# --- 3. Z-score Analysis (on Turnout_Percentage) ---
print(f"\n--- Z-score Analysis on 'Turnout_Percentage' (Threshold: {Z_SCORE_THRESHOLD}) ---")
results_summary = {} # To store all results

# Standard Z-score
turnout_values = df_analysis['Turnout_Percentage'].dropna()
if not turnout_values.empty:
    df_analysis['Turnout_Zscore'] = np.abs(stats.zscore(turnout_values))
    zscore_outliers = df_analysis[df_analysis['Turnout_Zscore'] > Z_SCORE_THRESHOLD]
    print(f"Found {len(zscore_outliers)} outliers using standard Z-score.")
    if not zscore_outliers.empty:
        zscore_outliers_path = os.path.join(OUTLIER_CLUSTER_OUTPUT_DIR, "zscore_turnout_outliers.csv")
        zscore_outliers[['URL_ID', 'Turnout_Percentage', 'Turnout_Zscore']].to_csv(zscore_outliers_path, index=False)
        print(f"Standard Z-score outliers saved to: {zscore_outliers_path}")
    results_summary['zscore_outliers_count'] = len(zscore_outliers)

    # Modified Z-score (more robust to outliers)
    median_turnout = np.median(turnout_values)
    mad_turnout = stats.median_abs_deviation(turnout_values, scale='normal') # scale='normal' makes it consistent with std for normal data

    if mad_turnout > 0: # Avoid division by zero if all values are the same
        df_analysis['Turnout_Modified_Zscore'] = np.abs(0.6745 * (turnout_values - median_turnout) / mad_turnout)
        mod_zscore_outliers = df_analysis[df_analysis['Turnout_Modified_Zscore'] > MODIFIED_Z_SCORE_THRESHOLD]
        print(f"Found {len(mod_zscore_outliers)} outliers using modified Z-score (Threshold: {MODIFIED_Z_SCORE_THRESHOLD}).")
        if not mod_zscore_outliers.empty:
            mod_zscore_outliers_path = os.path.join(OUTLIER_CLUSTER_OUTPUT_DIR, "modified_zscore_turnout_outliers.csv")
            mod_zscore_outliers[['URL_ID', 'Turnout_Percentage', 'Turnout_Modified_Zscore']].to_csv(mod_zscore_outliers_path, index=False)
            print(f"Modified Z-score outliers saved to: {mod_zscore_outliers_path}")
        results_summary['modified_zscore_outliers_count'] = len(mod_zscore_outliers)
    else:
        print("MAD is zero, cannot calculate Modified Z-score.")
        results_summary['modified_zscore_outliers_count'] = 0

    # Visualization for Z-score outliers
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=df_analysis.index, y='Turnout_Percentage', data=df_analysis, label='Normal Stations', s=30)
    if not zscore_outliers.empty:
        sns.scatterplot(x=zscore_outliers.index, y='Turnout_Percentage', data=zscore_outliers, color='red', label=f'Z-score Outliers (Z > {Z_SCORE_THRESHOLD})', s=70, marker='X')
    if mad_turnout > 0 and not mod_zscore_outliers.empty:
         sns.scatterplot(x=mod_zscore_outliers.index, y='Turnout_Percentage', data=mod_zscore_outliers, color='orange', label=f'Modified Z Outliers (MZ > {MODIFIED_Z_SCORE_THRESHOLD})', s=70, marker='P')
    plt.title('Turnout Percentage with Z-score Outliers Marked')
    plt.xlabel('Station Index (Arbitrary)')
    plt.ylabel('Turnout Percentage (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    zscore_plot_path = os.path.join(OUTLIER_CLUSTER_OUTPUT_DIR, "turnout_zscore_outliers_plot.png")
    plt.savefig(zscore_plot_path)
    print(f"Z-score outlier plot saved to: {zscore_plot_path}")
    plt.show()
else:
    print("No 'Turnout_Percentage' data to analyze with Z-scores.")
    results_summary['zscore_outliers_count'] = 0
    results_summary['modified_zscore_outliers_count'] = 0


# --- 4. Isolation Forest ---
print(f"\n--- Isolation Forest Analysis (Contamination: {ISO_FOREST_CONTAMINATION}) ---")
print(f"Using features: {current_mv_features}")

if not df_analysis_scaled[current_mv_features].empty:
    model_iso = IsolationForest(contamination=ISO_FOREST_CONTAMINATION, random_state=ISO_FOREST_RANDOM_STATE, n_jobs=-1)
    df_analysis['IsolationForest_Anomaly'] = model_iso.fit_predict(df_analysis_scaled[current_mv_features])
    # -1 indicates an outlier, 1 indicates an inlier

    iso_forest_outliers = df_analysis[df_analysis['IsolationForest_Anomaly'] == -1]
    print(f"Found {len(iso_forest_outliers)} outliers using Isolation Forest.")
    if not iso_forest_outliers.empty:
        iso_outliers_path = os.path.join(OUTLIER_CLUSTER_OUTPUT_DIR, "isolation_forest_outliers.csv")
        iso_forest_outliers[['URL_ID'] + current_mv_features].to_csv(iso_outliers_path, index=False)
        print(f"Isolation Forest outliers saved to: {iso_outliers_path}")
    results_summary['isolation_forest_outliers_count'] = len(iso_forest_outliers)

    # Visualization for Isolation Forest (using PCA if >2 features)
    if len(current_mv_features) > 0:
        pca_data = df_analysis_scaled[current_mv_features]
        title_suffix = ""
        if len(current_mv_features) > 2:
            pca = PCA(n_components=2, random_state=ISO_FOREST_RANDOM_STATE)
            pca_data = pca.fit_transform(df_analysis_scaled[current_mv_features])
            title_suffix = " (PCA Reduced)"
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}, Sum: {sum(pca.explained_variance_ratio_)}")

        plt.figure(figsize=(12, 7))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df_analysis['IsolationForest_Anomaly'], cmap='coolwarm', s=30, alpha=0.7)
        plt.title(f'Isolation Forest Outliers{title_suffix}')
        if len(current_mv_features) > 2 :
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
        elif len(current_mv_features) == 2:
            plt.xlabel(current_mv_features[0] + " (Scaled)")
            plt.ylabel(current_mv_features[1] + " (Scaled)")
        elif len(current_mv_features) == 1:
             plt.scatter(df_analysis_scaled[current_mv_features[0]], np.zeros_like(df_analysis_scaled[current_mv_features[0]]),
                        c=df_analysis['IsolationForest_Anomaly'], cmap='coolwarm', s=30, alpha=0.7)
             plt.xlabel(current_mv_features[0] + " (Scaled)")
             plt.yticks([])


        plt.colorbar(label='Anomaly (-1: Outlier, 1: Inlier)')
        plt.grid(True, linestyle='--', alpha=0.7)
        iso_plot_path = os.path.join(OUTLIER_CLUSTER_OUTPUT_DIR, "isolation_forest_outliers_plot.png")
        plt.savefig(iso_plot_path)
        print(f"Isolation Forest outlier plot saved to: {iso_plot_path}")
        plt.show()
else:
    print("Not enough data or features for Isolation Forest.")
    results_summary['isolation_forest_outliers_count'] = 0


# --- 5. DBSCAN Clustering ---
# DBSCAN can be sensitive to parameters 'eps' and 'min_samples'.
# Finding good parameters often requires experimentation (e.g., using k-distance plots for eps).
print(f"\n--- DBSCAN Clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}) ---")
print(f"Using features: {current_mv_features}")
print("Note: DBSCAN is sensitive to parameters. Results may vary with different eps/min_samples.")

if not df_analysis_scaled[current_mv_features].empty:
    model_dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    df_analysis['DBSCAN_Cluster'] = model_dbscan.fit_predict(df_analysis_scaled[current_mv_features])
    # Cluster label -1 indicates noise/outliers by DBSCAN

    dbscan_outliers = df_analysis[df_analysis['DBSCAN_Cluster'] == -1]
    n_clusters_ = len(set(df_analysis['DBSCAN_Cluster'])) - (1 if -1 in df_analysis['DBSCAN_Cluster'] else 0)
    print(f"Estimated number of clusters (excl. noise): {n_clusters_}")
    print(f"Found {len(dbscan_outliers)} noise points (outliers) using DBSCAN.")

    if not dbscan_outliers.empty:
        dbscan_outliers_path = os.path.join(OUTLIER_CLUSTER_OUTPUT_DIR, "dbscan_outliers.csv")
        dbscan_outliers[['URL_ID'] + current_mv_features].to_csv(dbscan_outliers_path, index=False)
        print(f"DBSCAN outliers saved to: {dbscan_outliers_path}")
    results_summary['dbscan_outliers_count'] = len(dbscan_outliers)
    results_summary['dbscan_num_clusters'] = n_clusters_

    # Visualization for DBSCAN (using PCA if >2 features)
    if len(current_mv_features) > 0:
        pca_data_dbscan = df_analysis_scaled[current_mv_features]
        title_suffix_dbscan = ""
        if len(current_mv_features) > 2:
            # Use a new PCA fit or the same if features are identical
            pca_dbscan = PCA(n_components=2, random_state=ISO_FOREST_RANDOM_STATE)
            pca_data_dbscan = pca_dbscan.fit_transform(df_analysis_scaled[current_mv_features])
            title_suffix_dbscan = " (PCA Reduced)"
            print(f"PCA for DBSCAN explained variance ratio: {pca_dbscan.explained_variance_ratio_}, Sum: {sum(pca_dbscan.explained_variance_ratio_)}")

        plt.figure(figsize=(12, 7))
        unique_labels = set(df_analysis['DBSCAN_Cluster'])
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1] # Black for noise
            class_member_mask = (df_analysis['DBSCAN_Cluster'] == k)
            xy = pca_data_dbscan[class_member_mask] if len(current_mv_features) > 2 else df_analysis_scaled[current_mv_features][class_member_mask].values

            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k' if k!= -1 else 'none', markersize=7 if k != -1 else 10, label=f'Cluster {k}' if k != -1 else 'Noise/Outlier')

        plt.title(f'DBSCAN Clustering Results{title_suffix_dbscan}')
        if len(current_mv_features) > 2 :
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
        elif len(current_mv_features) == 2:
            plt.xlabel(current_mv_features[0] + " (Scaled)")
            plt.ylabel(current_mv_features[1] + " (Scaled)")
        elif len(current_mv_features) == 1:
             # For 1D, scatter along y=0
            plt.scatter(df_analysis_scaled[current_mv_features[0]][df_analysis['DBSCAN_Cluster'] != -1],
                        np.zeros_like(df_analysis_scaled[current_mv_features[0]][df_analysis['DBSCAN_Cluster'] != -1]),
                        c=df_analysis['DBSCAN_Cluster'][df_analysis['DBSCAN_Cluster'] != -1], cmap='Spectral', s=30, alpha=0.7)
            plt.scatter(df_analysis_scaled[current_mv_features[0]][df_analysis['DBSCAN_Cluster'] == -1],
                        np.zeros_like(df_analysis_scaled[current_mv_features[0]][df_analysis['DBSCAN_Cluster'] == -1]),
                        color='black', marker='x', s=50, label='Noise/Outlier')

            plt.xlabel(current_mv_features[0] + " (Scaled)")
            plt.yticks([])


        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.7)
        dbscan_plot_path = os.path.join(OUTLIER_CLUSTER_OUTPUT_DIR, "dbscan_clusters_plot.png")
        plt.savefig(dbscan_plot_path)
        print(f"DBSCAN cluster plot saved to: {dbscan_plot_path}")
        plt.show()
else:
    print("Not enough data or features for DBSCAN.")
    results_summary['dbscan_outliers_count'] = 0
    results_summary['dbscan_num_clusters'] = 0


# --- 6. Save Summary ---
summary_file_path = os.path.join(OUTLIER_CLUSTER_OUTPUT_DIR, "outlier_cluster_summary.json")
with open(summary_file_path, 'w') as f:
    json.dump(results_summary, f, indent=4)
print(f"\nSummary of outlier detection saved to: {summary_file_path}")

print("\n--- Outlier & Cluster Detection Script Complete ---")