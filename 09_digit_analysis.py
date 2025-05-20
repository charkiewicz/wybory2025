import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
import os
import json

# --- 0. Configuration ---
# Input file with candidate-level votes
CANDIDATE_VOTES_FILE = 'data/polska_prezydent2025_obkw_kandydaci_NATIONAL_FINAL.csv'

# Output directory (can be the same as before or a new one)
LAST_DIGIT_ANALYSIS_OUTPUT_DIR = "data/last_digit_analysis_results"
os.makedirs(LAST_DIGIT_ANALYSIS_OUTPUT_DIR, exist_ok=True)

# Optional: Minimum number of vote entries for a candidate to be analyzed individually
MIN_VOTES_FOR_CANDIDATE_ANALYSIS = 100 # Adjust as needed

# --- 1. Load Data ---
print("--- Loading Candidate Votes Data ---")
try:
    df_votes = pd.read_csv(CANDIDATE_VOTES_FILE)
    print(f"Successfully loaded candidate votes data: {CANDIDATE_VOTES_FILE}")
    print(f"Shape: {df_votes.shape}")
    print(df_votes.head())
except FileNotFoundError:
    print(f"ERROR: File not found: {CANDIDATE_VOTES_FILE}. Please ensure the path is correct.")
    exit()
except Exception as e:
    print(f"Error loading {CANDIDATE_VOTES_FILE}: {e}")
    exit()

if 'Votes' not in df_votes.columns:
    print("ERROR: 'Votes' column not found in the CSV. This column is essential for last digit analysis.")
    exit()

# --- 2. Helper Function to Get Last Digit ---
def get_last_digit(n):
    """Extracts the last digit of an integer."""
    if pd.isna(n):
        return None
    return int(abs(n) % 10)

# --- 3. Perform Last Digit Analysis ---
def analyze_last_digits(vote_series, title_prefix="Overall", output_dir=LAST_DIGIT_ANALYSIS_OUTPUT_DIR):
    """
    Analyzes the frequency of last digits in a pandas Series of vote counts.

    Args:
        vote_series (pd.Series): Series containing vote counts.
        title_prefix (str): Prefix for plot titles and filenames.
        output_dir (str): Directory to save plots and results.

    Returns:
        dict: A dictionary containing analysis results.
    """
    print(f"\n--- Performing Last Digit Analysis for: {title_prefix} ---")
    results = {
        'title': title_prefix,
        'total_vote_entries_analyzed': 0,
        'observed_counts': {},
        'observed_frequencies': {},
        'expected_frequency_per_digit': 0.10, # Uniform distribution
        'chi_squared_statistic': None,
        'chi_squared_p_value': None,
        'chi_squared_conclusion': None,
        'plot_path': None
    }

    # Extract last digits from the 'Votes' column
    # We include 0 votes, as a pattern of 0s could also be interesting.
    last_digits = vote_series.apply(get_last_digit).dropna().astype(int)

    if last_digits.empty:
        print("No valid vote numbers to analyze for last digits.")
        results['error'] = "No valid vote numbers to analyze."
        return results

    results['total_vote_entries_analyzed'] = len(last_digits)
    print(f"Total vote entries analyzed: {len(last_digits)}")

    # Calculate observed frequencies
    digit_counts = Counter(last_digits)
    observed_counts_list = [digit_counts.get(d, 0) for d in range(10)]
    results['observed_counts'] = {d: digit_counts.get(d,0) for d in range(10)}

    observed_frequencies = {digit: count / len(last_digits) for digit, count in digit_counts.items()}
    results['observed_frequencies'] = {d: observed_frequencies.get(d,0.0) for d in range(10)}


    # Expected frequencies (uniform distribution)
    expected_counts_per_digit = len(last_digits) / 10.0
    expected_counts_list = [expected_counts_per_digit] * 10

    print("\nLast Digit Frequencies:")
    print("Digit | Obs. Count | Obs. Freq. | Exp. Freq.")
    print("------|------------|------------|-----------")
    for d in range(10):
        obs_c = digit_counts.get(d, 0)
        obs_f = observed_frequencies.get(d, 0.0)
        print(f"{d:^6}| {obs_c:^10} | {obs_f:^10.4f} | {0.10:^9.4f}")

    # Chi-squared goodness-of-fit test
    # H0: The observed distribution of last digits is uniform (as expected).
    # H1: The observed distribution of last digits is NOT uniform.
    if len(last_digits) >= 10 and all(ec >= 5 for ec in expected_counts_list): # Basic check for test validity
        chi2_stat, p_value = stats.chisquare(f_obs=observed_counts_list, f_exp=expected_counts_list)
        results['chi_squared_statistic'] = chi2_stat
        results['chi_squared_p_value'] = p_value
        print(f"\nChi-squared Test (Goodness of Fit to Uniform Distribution):")
        print(f"Chi2 Statistic: {chi2_stat:.4f}, P-value: {p_value:.4f}")
        alpha = 0.05
        if p_value < alpha:
            conclusion = f"P-value < {alpha}. The observed last digit distribution significantly deviates from a uniform distribution."
            print(conclusion)
        else:
            conclusion = f"P-value >= {alpha}. Cannot reject the null hypothesis; the observed distribution is consistent with a uniform distribution."
            print(conclusion)
        results['chi_squared_conclusion'] = conclusion
    else:
        print("\nChi-squared test not performed (or may be unreliable) due to small sample size or low expected counts per digit.")
        results['chi_squared_conclusion'] = "Test not performed/unreliable due to sample size/expected counts."


    # Visualization
    plt.figure(figsize=(12, 7))
    digits_for_plot = list(range(10))
    observed_freq_plot = [observed_frequencies.get(d, 0.0) for d in digits_for_plot]

    bars = plt.bar(digits_for_plot, observed_freq_plot, color='skyblue', label='Observed Frequency')
    plt.axhline(y=0.10, color='red', linestyle='--', label='Expected Frequency (10%)')

    plt.xlabel('Last Digit')
    plt.ylabel('Frequency')
    plt.title(f'Last Digit Frequency Distribution - {title_prefix}')
    plt.xticks(digits_for_plot)
    plt.ylim(0, max(0.15, max(observed_freq_plot) * 1.1) if observed_freq_plot else 0.15) # Adjust y-limit
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f'{yval:.3f}', ha='center', va='bottom')

    plot_filename = f"last_digit_distribution_{title_prefix.lower().replace(' ', '_').replace('/', '_')}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    results['plot_path'] = plot_path
    print(f"Last digit distribution plot saved to: {plot_path}")
    plt.show()

    return results

# --- 4. Run Analysis ---
all_analysis_results = []

# a) Overall analysis for all vote entries
overall_results = analyze_last_digits(
    df_votes['Votes'],
    title_prefix="Overall All Votes",
    output_dir=LAST_DIGIT_ANALYSIS_OUTPUT_DIR # Save directly in the main output dir
)
all_analysis_results.append(overall_results)

# b) Per-candidate analysis (optional)
if 'Candidate' in df_votes.columns:
    print("\n--- Performing Last Digit Analysis Per Candidate ---")
    unique_candidates = df_votes['Candidate'].unique()
    
    # Define and create the per_candidate subdirectory ONCE before the loop
    PER_CANDIDATE_SUBDIR = os.path.join(LAST_DIGIT_ANALYSIS_OUTPUT_DIR, "per_candidate")
    os.makedirs(PER_CANDIDATE_SUBDIR, exist_ok=True) # Ensure it exists
    
    for candidate in unique_candidates:
        candidate_votes = df_votes[df_votes['Candidate'] == candidate]['Votes']
        if len(candidate_votes.dropna()) >= MIN_VOTES_FOR_CANDIDATE_ANALYSIS:
            # Sanitize candidate name for filename (remove spaces, slashes etc.)
            sanitized_candidate_name = candidate.replace(' ', '_').replace('/', '_').replace('\\', '_')
            
            candidate_results = analyze_last_digits(
                candidate_votes,
                title_prefix=f"Candidate_{sanitized_candidate_name}",
                output_dir=PER_CANDIDATE_SUBDIR # Pass the now existing subdir
            )
            all_analysis_results.append(candidate_results)
        else:
            print(f"Skipping candidate '{candidate}' due to insufficient vote entries ({len(candidate_votes.dropna())} < {MIN_VOTES_FOR_CANDIDATE_ANALYSIS}).")

# --- 5. Save Summary of Results ---
summary_json_path = os.path.join(LAST_DIGIT_ANALYSIS_OUTPUT_DIR, "last_digit_analysis_summary.json")
with open(summary_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_analysis_results, f, ensure_ascii=False, indent=4)
print(f"\nJSON summary of all analyses saved to: {summary_json_path}")

# You can also create a more readable text summary if desired
summary_txt_path = os.path.join(LAST_DIGIT_ANALYSIS_OUTPUT_DIR, "last_digit_analysis_summary.txt")
with open(summary_txt_path, 'w', encoding='utf-8') as f:
    for res_item in all_analysis_results:
        f.write(f"--- Analysis for: {res_item.get('title', 'N/A')} ---\n")
        if res_item.get('error'):
            f.write(f"Error: {res_item['error']}\n\n")
            continue
        f.write(f"Total vote entries analyzed: {res_item.get('total_vote_entries_analyzed', 'N/A')}\n")
        f.write("Observed Frequencies:\n")
        for d in range(10):
            f.write(f"  Digit {d}: {res_item.get('observed_frequencies', {}).get(d, 0.0):.4f} (Count: {res_item.get('observed_counts', {}).get(d, 0)})\n")
        f.write(f"Expected Frequency per Digit: {res_item.get('expected_frequency_per_digit', 0.10):.2f}\n")
        f.write(f"Chi-squared Statistic: {res_item.get('chi_squared_statistic', 'N/A')}\n")
        f.write(f"Chi-squared P-value: {res_item.get('chi_squared_p_value', 'N/A')}\n")
        f.write(f"Chi-squared Conclusion: {res_item.get('chi_squared_conclusion', 'N/A')}\n")
        f.write(f"Plot saved to: {res_item.get('plot_path', 'N/A')}\n\n")

print(f"Text summary of all analyses saved to: {summary_txt_path}")
print("\n--- Last Digit Analysis Script Complete ---")