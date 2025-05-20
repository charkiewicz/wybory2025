import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os # For creating output directory
import json # For saving structured results (optional, could also be simple text)

# --- 0. Configuration for Saving Results ---
OUTPUT_DIR = "data/turnout_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create directory if it doesn't exist

# --- 1. Load Data ---
CSV_FILE_PATH = 'data/polska_prezydent2025_obkw_podsumowanie_NATIONAL_FINAL.csv'
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found. Please check the path and filename.")
    exit()
except Exception as e:
    print(f"Error loading CSV file '{CSV_FILE_PATH}': {e}")
    exit()

print("--- Initial Data Overview ---")
print(f"Shape of the DataFrame: {df.shape}")
print(df.head())

# Check if all required columns are present (simplified for this version as it seems correct now)
required_cols = ['URL_ID', 'liczba_uprawnionych', 'liczba_kart_wydanych_lacznie', 'liczba_glosow_waznych_na_kandydatow']
if not all(col in df.columns for col in required_cols):
    print(f"Error: CSV file {CSV_FILE_PATH} is missing one or more required columns: {required_cols}")
    exit()

print("\nRelevant columns data types:")
print(df[['URL_ID', 'liczba_uprawnionych', 'liczba_kart_wydanych_lacznie', 'liczba_glosow_waznych_na_kandydatow']].info())

# --- 2. Data Cleaning/Preparation ---
df_cleaned = df[df['liczba_uprawnionych'].notna() & (df['liczba_uprawnionych'] > 0)].copy()
if len(df_cleaned) < len(df):
    print(f"\nWarning: Removed {len(df) - len(df_cleaned)} rows due to missing or zero 'liczba_uprawnionych'.")

if df_cleaned.empty:
    print("Error: No valid data remaining after cleaning 'liczba_uprawnionych'. Cannot proceed.")
    exit()

# --- 3. Calculate Turnout per Voting Station ---
df_cleaned.loc[:, 'Turnout_Percentage'] = \
    (df_cleaned['liczba_kart_wydanych_lacznie'] / df_cleaned['liczba_uprawnionych']) * 100

# Save the DataFrame with Turnout_Percentage to a new CSV (optional)
df_cleaned.to_csv(os.path.join(OUTPUT_DIR, "data_with_turnout.csv"), index=False)
print(f"\nDataFrame with Turnout_Percentage saved to: {os.path.join(OUTPUT_DIR, 'data_with_turnout.csv')}")


print("\n--- Turnout Calculation ---")
print(df_cleaned[['URL_ID', 'liczba_uprawnionych', 'liczba_kart_wydanych_lacznie', 'Turnout_Percentage']].head())

# --- 4. Statistical Tests on Turnout ---
results_summary = {} # Dictionary to store textual results

# Descriptive Statistics
print("\n--- Descriptive Statistics for Turnout Percentage ---")
desc_stats = df_cleaned['Turnout_Percentage'].describe()
print(desc_stats)
median_turnout = df_cleaned['Turnout_Percentage'].median()
print(f"Median Turnout: {median_turnout:.2f}%")

results_summary['descriptive_statistics'] = desc_stats.to_dict()
results_summary['descriptive_statistics']['median'] = median_turnout # Add median separately if not in describe() in the exact format

# Histogram or Density Plot
plt.figure(figsize=(12, 7)) # Slightly larger for better saving quality
sns.histplot(df_cleaned['Turnout_Percentage'], kde=True, bins=50) # More bins for a larger dataset
plt.title('Distribution of Turnout Percentage per Voting Station')
plt.xlabel('Turnout Percentage (%)')
plt.ylabel('Frequency (Number of Voting Stations)')
plt.grid(True, linestyle='--', alpha=0.7)
histogram_path = os.path.join(OUTPUT_DIR, "turnout_distribution_histogram.png")
plt.savefig(histogram_path)
print(f"Turnout distribution histogram saved to: {histogram_path}")
plt.show() # Still show it interactively

# Shapiro-Wilk Test for Normality
print("\n--- Normality Test for Turnout Percentage (Shapiro-Wilk) ---")
results_summary['shapiro_wilk_test'] = {}
turnout_data_for_normality = df_cleaned['Turnout_Percentage'].dropna()

# For very large N, Shapiro-Wilk's p-value can be unreliable or the test slow.
# It's often recommended for N < 5000. We can take a sample if N is too large.
sample_for_shapiro = turnout_data_for_normality
shapiro_notes = ""
if len(turnout_data_for_normality) > 5000:
    shapiro_notes = "Note: For N > 5000, computed p-value may be less accurate or test slow. Using a random sample of 5000."
    print(shapiro_notes)
    sample_for_shapiro = turnout_data_for_normality.sample(5000, random_state=42) # random_state for reproducibility
    
if len(sample_for_shapiro) >= 3: # Shapiro-Wilk needs at least 3 data points
    shapiro_stat, shapiro_p_value = stats.shapiro(sample_for_shapiro)
    print(f"Shapiro-Wilk Statistic (on sample of {len(sample_for_shapiro)}): {shapiro_stat:.4f}")
    print(f"P-value: {shapiro_p_value:.4f}")
    alpha = 0.05
    shapiro_conclusion = "Sample looks normally distributed (fail to reject H0)" if shapiro_p_value > alpha else "Sample does not look normally distributed (reject H0)"
    print(shapiro_conclusion)
    results_summary['shapiro_wilk_test'] = {
        'notes': shapiro_notes,
        'sample_size_used': len(sample_for_shapiro),
        'statistic': shapiro_stat,
        'p_value': shapiro_p_value,
        'conclusion (alpha=0.05)': shapiro_conclusion
    }
else:
    print("Not enough data points to perform Shapiro-Wilk test.")
    results_summary['shapiro_wilk_test'] = {'error': 'Not enough data points'}

# Kolmogorov-Smirnov Test for Normality
print("\n--- Normality Test for Turnout Percentage (Kolmogorov-Smirnov) ---")
results_summary['kolmogorov_smirnov_test'] = {}
if len(turnout_data_for_normality) >= 2: # KS test needs at least 2 data points
    ks_stat, ks_p_value = stats.kstest(turnout_data_for_normality, 'norm',
                                       args=(turnout_data_for_normality.mean(), turnout_data_for_normality.std()))
    print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.4f}")
    print(f"P-value: {ks_p_value:.4f}")
    ks_conclusion = "Sample looks normally distributed (fail to reject H0)" if ks_p_value > alpha else "Sample does not look normally distributed (reject H0)"
    print(ks_conclusion)
    results_summary['kolmogorov_smirnov_test'] = {
        'statistic': ks_stat,
        'p_value': ks_p_value,
        'conclusion (alpha=0.05)': ks_conclusion
    }
else:
    print("Not enough data points to perform Kolmogorov-Smirnov test.")
    results_summary['kolmogorov_smirnov_test'] = {'error': 'Not enough data points'}


# --- 5. Additional Analysis (Optional) ---

# Scatter plot of turnout versus total valid votes
plt.figure(figsize=(12, 7))
sns.scatterplot(x='liczba_glosow_waznych_na_kandydatow', y='Turnout_Percentage', data=df_cleaned, alpha=0.5, s=15) # s for point size
plt.title('Turnout Percentage vs. Total Valid Votes per Station')
plt.xlabel('Total Valid Votes for Candidates')
plt.ylabel('Turnout Percentage (%)')
plt.grid(True, linestyle='--', alpha=0.7)
scatter_plot_path = os.path.join(OUTPUT_DIR, "turnout_vs_valid_votes_scatter.png")
plt.savefig(scatter_plot_path)
print(f"Scatter plot saved to: {scatter_plot_path}")
plt.show() # Still show it interactively

# --- 6. Save Textual Results ---
summary_file_path = os.path.join(OUTPUT_DIR, "turnout_analysis_summary.txt")
with open(summary_file_path, 'w', encoding='utf-8') as f:
    f.write("--- Turnout Analysis Summary ---\n\n")

    f.write("=== Data Overview ===\n")
    f.write(f"Input CSV: {CSV_FILE_PATH}\n")
    f.write(f"Initial shape of DataFrame: {df.shape}\n")
    f.write(f"Shape after cleaning (valid 'liczba_uprawnionych'): {df_cleaned.shape}\n")
    f.write(f"Number of rows removed during cleaning: {len(df) - len(df_cleaned)}\n\n")

    f.write("=== Descriptive Statistics for Turnout Percentage ===\n")
    f.write(desc_stats.to_string()) # Convert pandas Series to string
    f.write(f"\nMedian Turnout: {median_turnout:.2f}%\n\n")

    f.write("=== Normality Test for Turnout Percentage (Shapiro-Wilk) ===\n")
    if 'error' in results_summary['shapiro_wilk_test']:
        f.write(f"{results_summary['shapiro_wilk_test']['error']}\n")
    else:
        if results_summary['shapiro_wilk_test']['notes']:
             f.write(f"{results_summary['shapiro_wilk_test']['notes']}\n")
        f.write(f"Sample size used for test: {results_summary['shapiro_wilk_test']['sample_size_used']}\n")
        f.write(f"Statistic: {results_summary['shapiro_wilk_test']['statistic']:.4f}\n")
        f.write(f"P-value: {results_summary['shapiro_wilk_test']['p_value']:.4f}\n")
        f.write(f"Conclusion (alpha=0.05): {results_summary['shapiro_wilk_test']['conclusion (alpha=0.05)']}\n\n")

    f.write("=== Normality Test for Turnout Percentage (Kolmogorov-Smirnov) ===\n")
    if 'error' in results_summary['kolmogorov_smirnov_test']:
        f.write(f"{results_summary['kolmogorov_smirnov_test']['error']}\n")
    else:
        f.write(f"Statistic: {results_summary['kolmogorov_smirnov_test']['statistic']:.4f}\n")
        f.write(f"P-value: {results_summary['kolmogorov_smirnov_test']['p_value']:.4f}\n")
        f.write(f"Conclusion (alpha=0.05): {results_summary['kolmogorov_smirnov_test']['conclusion (alpha=0.05)']}\n\n")

    f.write("=== Saved Files ===\n")
    f.write(f"Cleaned data with turnout: {os.path.join(OUTPUT_DIR, 'data_with_turnout.csv')}\n")
    f.write(f"Turnout distribution histogram: {histogram_path}\n")
    f.write(f"Turnout vs Valid Votes scatter plot: {scatter_plot_path}\n")

# Optional: Save as JSON for more structured output
summary_json_path = os.path.join(OUTPUT_DIR, "turnout_analysis_summary.json")
with open(summary_json_path, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=4)

print(f"\nTextual summary saved to: {summary_file_path}")
print(f"JSON summary saved to: {summary_json_path}")
print("\n--- Analysis Complete ---")

# Geographical Visualization (Commented out)
print("\nGeographical visualization requires additional data (lat/lon) and libraries (e.g., GeoPandas, Folium).")
print("This part is commented out in the script.")