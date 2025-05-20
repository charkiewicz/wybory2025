import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
import os
import json

# --- 0. Configuration ---
BASE_OUTPUT_DIR = "data/turnout_analysis_results"
INPUT_DATA_FILE = os.path.join(BASE_OUTPUT_DIR, "data_with_turnout.csv")
ADVANCED_ANALYSIS_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "advanced_anomaly_detection")
os.makedirs(ADVANCED_ANALYSIS_OUTPUT_DIR, exist_ok=True)

PCA_N_COMPONENTS = 0.95
PCA_RANDOM_STATE = 42
OCSVM_NU = 0.05
OCSVM_KERNEL = 'rbf'
OCSVM_GAMMA = 'auto'
AE_ENCODING_DIM_RATIO = 0.5
AE_EPOCHS = 100
# AE_BATCH_SIZE = 64 # Not directly used by MLPRegressor in this simple setup, but kept for consistency if expanding
AE_VALIDATION_SPLIT = 0.1
AE_EARLY_STOPPING_PATIENCE = 10
AE_OUTLIER_THRESHOLD_PERCENTILE = 95

MULTIVARIATE_FEATURES = [
    'Turnout_Percentage', 'liczba_uprawnionych', 'liczba_kart_wydanych_lacznie',
    'liczba_glosow_waznych_na_kandydatow', 'liczba_kart_niewykorzystanych',
    'liczba_kart_niewaznych', 'liczba_glosow_niewaznych'
]

# --- 1. Load Data ---
print("--- Loading Data ---")
try:
    df = pd.read_csv(INPUT_DATA_FILE)
    print(f"Successfully loaded data: {INPUT_DATA_FILE} (Shape: {df.shape})")
except FileNotFoundError:
    print(f"ERROR: File not found: {INPUT_DATA_FILE}. Please run script 7 first.")
    exit()
except Exception as e:
    print(f"Error loading {INPUT_DATA_FILE}: {e}")
    exit()

# --- 2. Data Preparation ---
df_analysis = df.copy()
results_summary = {}

actual_mv_features = [f for f in MULTIVARIATE_FEATURES if f in df_analysis.columns]
if not actual_mv_features:
    print("ERROR: None of the specified MULTIVARIATE_FEATURES are present.")
    exit()
if len(actual_mv_features) < len(MULTIVARIATE_FEATURES):
    print(f"Warning: Using subset of features: {actual_mv_features}")
print(f"Using features for advanced analysis: {actual_mv_features}")

df_analysis.dropna(subset=actual_mv_features, inplace=True)
print(f"Shape after dropping NaNs: {df_analysis.shape}")

if df_analysis.empty or len(df_analysis) < 10:
    print("Not enough data after preprocessing. Cannot proceed.")
    exit()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_analysis[actual_mv_features])
# df_X_scaled = pd.DataFrame(X_scaled, columns=actual_mv_features, index=df_analysis.index) # Not strictly needed if X_scaled is used directly

# --- 3. Principal Component Analysis (PCA) ---
print(f"\n--- Principal Component Analysis (PCA) ---")
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=PCA_RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled) # X_pca will be used for visualizations

print(f"PCA: {pca.n_components_} components explain {np.sum(pca.explained_variance_ratio_):.4f} variance.")
results_summary['pca'] = {
    'n_components_selected': pca.n_components_,
    'explained_variance_ratio_per_component': pca.explained_variance_ratio_.tolist(),
    'total_explained_variance': np.sum(pca.explained_variance_ratio_)
}

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=20)
plt.title('PCA: First Two Principal Components')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
plt.grid(True, linestyle='--', alpha=0.7)
pca_plot_path = os.path.join(ADVANCED_ANALYSIS_OUTPUT_DIR, "pca_first_two_components.png")
plt.savefig(pca_plot_path)
print(f"PCA plot saved to: {pca_plot_path}")
plt.show()

# --- 4. One-Class SVM ---
print(f"\n--- One-Class SVM (nu={OCSVM_NU}, kernel='{OCSVM_KERNEL}') ---")
ocsvm = OneClassSVM(nu=OCSVM_NU, kernel=OCSVM_KERNEL, gamma=OCSVM_GAMMA)
df_analysis['OCSVM_Anomaly'] = ocsvm.fit_predict(X_scaled) # -1 for outliers

ocsvm_outliers = df_analysis[df_analysis['OCSVM_Anomaly'] == -1]
print(f"Found {len(ocsvm_outliers)} outliers using One-Class SVM.")
if not ocsvm_outliers.empty:
    ocsvm_outliers_path = os.path.join(ADVANCED_ANALYSIS_OUTPUT_DIR, "one_class_svm_outliers.csv")
    ocsvm_outliers[['URL_ID'] + actual_mv_features].to_csv(ocsvm_outliers_path, index=False)
    print(f"One-Class SVM outliers saved to: {ocsvm_outliers_path}")
results_summary['one_class_svm'] = {'outliers_count': len(ocsvm_outliers)}

plt.figure(figsize=(10, 6))
# Use X_pca for consistent visualization if available and appropriate
plot_x_ocsvm, plot_y_ocsvm = X_pca[:, 0], X_pca[:, 1]
plot_title_ocsvm = 'One-Class SVM Outliers (PCA Reduced)'
xlabel_ocsvm, ylabel_ocsvm = 'PC1', 'PC2'

if X_scaled.shape[1] == 1:
    plot_x_ocsvm, plot_y_ocsvm = X_scaled[:,0], np.zeros_like(X_scaled[:,0])
    plot_title_ocsvm, xlabel_ocsvm, ylabel_ocsvm = 'One-Class SVM Outliers', actual_mv_features[0] + " (Scaled)", ""
elif X_scaled.shape[1] == 2:
    plot_x_ocsvm, plot_y_ocsvm = X_scaled[:,0], X_scaled[:,1]
    plot_title_ocsvm = 'One-Class SVM Outliers'
    xlabel_ocsvm, ylabel_ocsvm = actual_mv_features[0] + " (Scaled)", actual_mv_features[1] + " (Scaled)"

plt.scatter(plot_x_ocsvm[df_analysis['OCSVM_Anomaly'] == 1], plot_y_ocsvm[df_analysis['OCSVM_Anomaly'] == 1],
            label='Inlier', c='skyblue', alpha=0.7, s=30)
plt.scatter(plot_x_ocsvm[df_analysis['OCSVM_Anomaly'] == -1], plot_y_ocsvm[df_analysis['OCSVM_Anomaly'] == -1],
            label='Outlier (OCSVM)', c='red', marker='x', s=70)
plt.title(plot_title_ocsvm)
plt.xlabel(xlabel_ocsvm)
if ylabel_ocsvm: plt.ylabel(ylabel_ocsvm)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
ocsvm_plot_path = os.path.join(ADVANCED_ANALYSIS_OUTPUT_DIR, "one_class_svm_outliers_plot.png")
plt.savefig(ocsvm_plot_path)
print(f"One-Class SVM outlier plot saved to: {ocsvm_plot_path}")
plt.show()

# --- 5. Autoencoder (using Scikit-learn MLPRegressor) ---
print(f"\n--- Autoencoder Anomaly Detection (using Scikit-learn MLPRegressor) ---")
input_dim_sklearn = X_scaled.shape[1]
hidden_layer_size_sklearn = max(1, int(input_dim_sklearn * AE_ENCODING_DIM_RATIO))
print(f"Sklearn Autoencoder: Input Dim={input_dim_sklearn}, Hidden Layer Size={hidden_layer_size_sklearn}")

autoencoder_sklearn = MLPRegressor(
    hidden_layer_sizes=(hidden_layer_size_sklearn,), activation='relu', solver='adam',
    alpha=1e-4, learning_rate_init=0.001, max_iter=AE_EPOCHS,
    random_state=PCA_RANDOM_STATE, early_stopping=True,
    validation_fraction=AE_VALIDATION_SPLIT, n_iter_no_change=AE_EARLY_STOPPING_PATIENCE,
    verbose=False
)

print("Training Sklearn Autoencoder...")
try:
    autoencoder_sklearn.fit(X_scaled, X_scaled)
except Exception as e:
    print(f"Error during Sklearn MLPRegressor training: {e}. Skipping Sklearn Autoencoder.")
    results_summary['sklearn_autoencoder'] = {'error': str(e), 'outliers_count': 0}
else:
    print("Sklearn Autoencoder training complete.")
    if hasattr(autoencoder_sklearn, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(autoencoder_sklearn.loss_curve_, label='Training Loss')
        if hasattr(autoencoder_sklearn, 'validation_scores_') and autoencoder_sklearn.validation_scores_ is not None:
            val_steps = autoencoder_sklearn.n_iter_no_change if autoencoder_sklearn.solver != 'lbfgs' else 1
            val_x = np.arange(len(autoencoder_sklearn.validation_scores_)) * val_steps
            val_x = val_x[val_x < len(autoencoder_sklearn.loss_curve_)]
            plt.plot(val_x, autoencoder_sklearn.validation_scores_[:len(val_x)], label='Validation Score', linestyle='--')
        plt.title('Sklearn Autoencoder Training History'); plt.xlabel('Epoch/Iteration'); plt.ylabel('Loss (MSE)')
        plt.legend(); plt.grid(True)
        sklearn_ae_history_plot_path = os.path.join(ADVANCED_ANALYSIS_OUTPUT_DIR, "sklearn_autoencoder_training_history.png")
        plt.savefig(sklearn_ae_history_plot_path)
        print(f"Sklearn Autoencoder training history plot saved to: {sklearn_ae_history_plot_path}")
        plt.show()

    X_pred_sklearn_ae = autoencoder_sklearn.predict(X_scaled)
    mse_sklearn_ae = np.mean(np.power(X_scaled - X_pred_sklearn_ae, 2), axis=1)
    df_analysis['Sklearn_AE_Reconstruction_Error'] = mse_sklearn_ae
    error_threshold_sklearn_ae = np.percentile(mse_sklearn_ae, AE_OUTLIER_THRESHOLD_PERCENTILE)
    print(f"Sklearn AE reconstruction error threshold ({AE_OUTLIER_THRESHOLD_PERCENTILE}th percentile): {error_threshold_sklearn_ae:.4f}")
    df_analysis['Sklearn_AE_Anomaly'] = (mse_sklearn_ae > error_threshold_sklearn_ae).astype(int)
    sklearn_ae_outliers = df_analysis[df_analysis['Sklearn_AE_Anomaly'] == 1]
    print(f"Found {len(sklearn_ae_outliers)} outliers using Sklearn Autoencoder.")
    if not sklearn_ae_outliers.empty:
        path = os.path.join(ADVANCED_ANALYSIS_OUTPUT_DIR, "sklearn_autoencoder_outliers.csv")
        sklearn_ae_outliers[['URL_ID'] + actual_mv_features + ['Sklearn_AE_Reconstruction_Error']].to_csv(path, index=False)
        print(f"Sklearn Autoencoder outliers saved to: {path}")
    results_summary['sklearn_autoencoder'] = {
        'outliers_count': len(sklearn_ae_outliers),
        'reconstruction_error_threshold': error_threshold_sklearn_ae,
        'threshold_percentile': AE_OUTLIER_THRESHOLD_PERCENTILE
    }

    plt.figure(figsize=(10, 6))
    sns.histplot(mse_sklearn_ae, bins=50, kde=True)
    plt.axvline(error_threshold_sklearn_ae, color='r', linestyle='--', label=f'{AE_OUTLIER_THRESHOLD_PERCENTILE}th Pctl Thresh')
    plt.title('Distribution of Sklearn Autoencoder Reconstruction Errors'); plt.xlabel('MSE'); plt.ylabel('Frequency')
    plt.legend()
    sklearn_ae_error_dist_path = os.path.join(ADVANCED_ANALYSIS_OUTPUT_DIR, "sklearn_autoencoder_reconstruction_errors.png")
    plt.savefig(sklearn_ae_error_dist_path)
    print(f"Sklearn Autoencoder error distribution plot saved to: {sklearn_ae_error_dist_path}")
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_x_ae, plot_y_ae = X_pca[:, 0], X_pca[:, 1]
    plot_title_ae = 'Sklearn Autoencoder Outliers (PCA Reduced)'
    xlabel_ae, ylabel_ae = 'PC1', 'PC2'
    if X_scaled.shape[1] == 1:
        plot_x_ae, plot_y_ae = X_scaled[:,0], np.zeros_like(X_scaled[:,0])
        plot_title_ae, xlabel_ae, ylabel_ae = 'Sklearn Autoencoder Outliers', actual_mv_features[0] + " (Scaled)", ""
    elif X_scaled.shape[1] == 2:
        plot_x_ae, plot_y_ae = X_scaled[:,0], X_scaled[:,1]
        plot_title_ae = 'Sklearn Autoencoder Outliers'
        xlabel_ae, ylabel_ae = actual_mv_features[0] + " (Scaled)", actual_mv_features[1] + " (Scaled)"

    plt.scatter(plot_x_ae[df_analysis['Sklearn_AE_Anomaly'] == 0], plot_y_ae[df_analysis['Sklearn_AE_Anomaly'] == 0],
                label='Inlier (Sklearn AE)', c='skyblue', alpha=0.7, s=30)
    plt.scatter(plot_x_ae[df_analysis['Sklearn_AE_Anomaly'] == 1], plot_y_ae[df_analysis['Sklearn_AE_Anomaly'] == 1],
                label='Outlier (Sklearn AE)', c='green', marker='D', s=70)
    plt.title(plot_title_ae); plt.xlabel(xlabel_ae)
    if ylabel_ae: plt.ylabel(ylabel_ae)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    sklearn_ae_outlier_plot_path = os.path.join(ADVANCED_ANALYSIS_OUTPUT_DIR, "sklearn_autoencoder_outliers_plot.png")
    plt.savefig(sklearn_ae_outlier_plot_path)
    print(f"Sklearn Autoencoder outlier plot saved to: {sklearn_ae_outlier_plot_path}")
    plt.show()

# --- 6. Save Summary ---
summary_file_path = os.path.join(ADVANCED_ANALYSIS_OUTPUT_DIR, "advanced_analysis_summary.json")
with open(summary_file_path, 'w') as f:
    # Ensure all values in results_summary are JSON serializable
    # The default lambda for ndarray is good. Check for other non-serializable types if errors persist.
    def custom_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Add more types if necessary
        # print(f"Warning: Cannot serialize object of type {type(obj)}") # For debugging
        # return str(obj) # Fallback, but might lose info
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    json.dump(results_summary, f, indent=4, default=custom_serializer)
print(f"\nSummary of advanced anomaly detection saved to: {summary_file_path}")

print("\n--- Advanced Anomaly Detection Script Complete ---")