import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import Counter
import math

# --- Configuration ---
CSV_FILE_PATH = 'data/polska_prezydent2025_obkw_kandydaci_NATIONAL_FINAL.csv' # Or your actual file path like 'data/polska_prezydent2025_obkw_kandydaci_NATIONAL_FINAL.csv'
OUTPUT_PLOT_DIR = 'data/benford_plots/'
MIN_DATA_POINTS_FOR_PLOTTING = 0 # Minimum number of first digits for meaningful plots

# --- Helper Functions ---
def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\._-]', '', name)
    return name

def get_first_digit(n):
    """Extracts the first significant digit of a positive number."""
    if pd.isna(n) or n <= 0:
        return None
    n_int = int(n)
    while n_int >= 10:
        n_int //= 10
    return n_int

def calculate_benford_data_for_plotting(votes_series):
    """
    Calculates observed and expected Benford proportions for a given series of votes.
    Returns: (observed_props_array, benford_expected_props_array, total_valid_digits, error_message)
    Proportions are for digits 1-9.
    """
    positive_votes = votes_series[votes_series > 0].dropna()
    if positive_votes.empty:
        return None, None, 0, "No positive vote entries found."

    first_digits = positive_votes.apply(get_first_digit).dropna().astype(int)
    total_valid_digits = len(first_digits)

    if total_valid_digits == 0:
        return np.zeros(9), np.array([math.log10(1 + 1/d) for d in range(1, 10)]), 0, "No valid first digits to analyze."

    if total_valid_digits < MIN_DATA_POINTS_FOR_PLOTTING:
        # We can still calculate, but the error message will indicate low data
        error_msg = (f"Warning: Low data points ({total_valid_digits}). "
                     f"Minimum recommended: {MIN_DATA_POINTS_FOR_PLOTTING}.")
    else:
        error_msg = None

    observed_counts = Counter(first_digits)
    observed_props_array = np.array([observed_counts.get(d, 0) / total_valid_digits for d in range(1, 10)])
    benford_expected_props_array = np.array([math.log10(1 + 1/d) for d in range(1, 10)])

    return observed_props_array, benford_expected_props_array, total_valid_digits, error_msg


def generate_benford_plots(observed_props, expected_props, total_digits, title_prefix, output_dir, filename_base):
    """
    Generates and saves two plots:
    1. Bar chart of observed vs. Benford expected first-digit proportions.
    2. CDF plot of observed vs. Benford expected cumulative first-digit proportions.
    """
    digits = np.arange(1, 10)

    # 1. Bar Chart: Observed vs. Expected Proportions
    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    plt.bar(digits - bar_width/2, observed_props, bar_width, label=f'Observed (N={total_digits})', color='skyblue')
    plt.bar(digits + bar_width/2, expected_props, bar_width, label='Benford Expected', color='salmon', alpha=0.7)

    plt.xlabel('First Digit')
    plt.ylabel('Proportion')
    plt.title(f'{title_prefix}\nFirst Digit Distribution: Observed vs. Benford Expected')
    plt.xticks(digits)
    plt.yticks(np.arange(0, max(np.max(observed_props), np.max(expected_props)) * 1.1 , 0.05))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path_bar = os.path.join(output_dir, f"{filename_base}_distribution_bar.png")
    plt.savefig(plot_path_bar)
    plt.close()
    print(f"  Saved bar chart: {plot_path_bar}")

    # 2. CDF Plot
    observed_cdf = np.cumsum(observed_props)
    expected_cdf = np.cumsum(expected_props)

    plt.figure(figsize=(10, 6))
    plt.plot(digits, observed_cdf, marker='o', linestyle='-', label=f'Observed CDF (N={total_digits})', color='blue')
    plt.plot(digits, expected_cdf, marker='x', linestyle='--', label='Benford Expected CDF', color='red')

    plt.xlabel('First Digit')
    plt.ylabel('Cumulative Proportion')
    plt.title(f'{title_prefix}\nCumulative Distribution: Observed vs. Benford Expected')
    plt.xticks(digits)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path_cdf = os.path.join(output_dir, f"{filename_base}_cdf_plot.png")
    plt.savefig(plot_path_cdf)
    plt.close()
    print(f"  Saved CDF plot: {plot_path_cdf}")


# --- Main script ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    print(f"Output directory for plots: '{OUTPUT_PLOT_DIR}'")

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded data from '{CSV_FILE_PATH}'.\n")
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found. Please check the path.")
        exit()

    # Ensure 'Votes' column is numeric
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

    # 1. Plots for ALL CANDIDATES COMBINED
    print("Processing: All Candidates Combined")
    all_votes_series = df['Votes'].copy()
    obs_props_all, exp_props_all, total_digits_all, error_all = calculate_benford_data_for_plotting(all_votes_series)

    if error_all and "No positive vote entries found." in error_all:
        print(f"  Skipping plots for All Candidates Combined: {error_all}")
    elif obs_props_all is not None and exp_props_all is not None:
        if error_all: # This means low data points warning
            print(f"  {error_all}")
        generate_benford_plots(obs_props_all, exp_props_all, total_digits_all,
                               "All Candidates Combined", OUTPUT_PLOT_DIR, "all_candidates_combined")
    else:
        print(f"  Skipping plots for All Candidates Combined due to error: {error_all}")
    print("-" * 50)


    # 2. Plots for EACH CANDIDATE SEPARATELY
    print("\nProcessing: Individual Candidates")
    unique_candidates = df['Candidate'].dropna().unique() # Drop NaN candidate names for iteration

    if len(unique_candidates) == 0:
        print("No unique candidates found to process individually.")
    else:
        for candidate_name in unique_candidates:
            print(f"Processing Candidate: {candidate_name}")
            candidate_votes_series = df[df['Candidate'] == candidate_name]['Votes'].copy()

            obs_props_cand, exp_props_cand, total_digits_cand, error_cand = \
                calculate_benford_data_for_plotting(candidate_votes_series)

            if error_cand and "No positive vote entries found." in error_cand:
                print(f"  Skipping plots for {candidate_name}: {error_cand}")
                continue
            if error_cand and f"Low data points ({total_digits_cand})" in error_cand:
                 print(f"  {error_cand} Plots will be generated but interpret with caution.")
            elif obs_props_cand is None or exp_props_cand is None:
                print(f"  Skipping plots for {candidate_name} due to error: {error_cand if error_cand else 'Unknown calculation error'}")
                continue

            # If we reach here, it means we have data (even if low) to plot
            s_candidate_name = sanitize_filename(str(candidate_name)) # Ensure candidate_name is str for sanitize
            generate_benford_plots(obs_props_cand, exp_props_cand, total_digits_cand,
                                   f"Candidate: {candidate_name}", OUTPUT_PLOT_DIR, s_candidate_name)
            print("-" * 30)

    print("\nAll plotting tasks complete.")