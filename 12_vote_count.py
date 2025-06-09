import pandas as pd
import os

def analyze_election_file(filepath):
    """
    Reads an election results CSV, calculates the total votes per candidate,
    and prints the results sorted by vote count.

    Args:
        filepath (str): The full path to the CSV file.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)

        # --- CORE LOGIC ---
        # Group the data by the 'Candidate' column.
        # For each candidate, sum up their 'Votes'.
        # The result is a new pandas Series.
        candidate_totals = df.groupby('Candidate')['Votes'].sum()

        # Sort the results in descending order for better readability
        sorted_totals = candidate_totals.sort_values(ascending=False)

        # --- PRINT RESULTS ---
        # Get the base filename for a clean title
        filename = os.path.basename(filepath)
        print(f"--- Vote Summary for: {filename} ---")
        print(sorted_totals)
        print("-" * 50 + "\n") # Add a separator for clarity

    except FileNotFoundError:
        print(f"!!! ERROR: The file was not found at '{filepath}'")
        print("Please ensure the file exists and the path is correct.\n")
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}\n")


# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    # Define the directory where the data is stored
    data_directory = 'data'

    # List of the files to be processed (with corrected filenames)
    files_to_process = [
        'polska_prezydent2025_obkw_kandydaci_NATIONAL_FINAL.csv',
        'polska_prezydent2025_tura2_obkw_kandydaci_NATIONAL_FINAL.csv'
    ]

    # Loop through each file and process it
    for file in files_to_process:
        # Construct the full path to the file
        full_path = os.path.join(data_directory, file)
        analyze_election_file(full_path)