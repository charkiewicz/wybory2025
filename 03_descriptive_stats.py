import pandas as pd
import numpy as np # For NaN if needed, though pandas handles it well
from scipy import stats as sp_stats # For a potentially more robust mode calculation

# --- Configuration ---
CSV_FILE_PATH = 'data/polska_prezydent2025_tura2_obkw_kandydaci_NATIONAL_FINAL.csv'
OUTPUT_STATS_FILE_PATH = 'data/candidate_descriptive_statistics.csv' # Output file

# --- Helper function for mode ---
def calculate_mode(series):
    """
    Calculates the mode(s) of a pandas Series.
    If multiple modes exist, they are returned as a comma-separated string.
    If no mode (e.g., all unique values and series is short), returns pd.NA.
    """
    mode_result = series.mode()
    if mode_result.empty:
        return pd.NA
    elif len(mode_result) == 1:
        return mode_result.iloc[0]
    else:
        # Convert modes to string and join, in case modes are numeric
        return ', '.join(map(str, mode_result.tolist()))

# --- Main script ---
if __name__ == "__main__":
    # 1. Load the CSV data
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded data from '{CSV_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        print("Please make sure the CSV file is in the correct location and the path is correct.")
        # Create a dummy CSV for testing if it's missing (optional, from previous script)
        print("\nCreating a dummy 'election_data.csv' for demonstration purposes...")
        dummy_data = """Wojewodztwo_Name,Powiat_MnpP_Name,Gmina_Name,URL_ID,Candidate,Votes
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BARTOSZEWICZ Artur,2
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BIEJAT Magdalena Agnieszka,32
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BIEJAT Magdalena Agnieszka,32
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,TRZASKOWSKI Rafa Kazimierz,361
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404409,BARTOSZEWICZ Artur,6
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404409,BIEJAT Magdalena Agnieszka,29
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404409,TRZASKOWSKI Rafa Kazimierz,381
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404500,BARTOSZEWICZ Artur,10
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404500,BIEJAT Magdalena Agnieszka,50
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404500,TRZASKOWSKI Rafa Kazimierz,400
"""
        with open(CSV_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(dummy_data)
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Dummy '{CSV_FILE_PATH}' created and loaded.")
        exit() # Exit if dummy was created, as the stats won't be meaningful

    # Ensure 'Votes' column is numeric
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    # Drop rows where 'Votes' could not be converted (if any)
    df.dropna(subset=['Votes'], inplace=True)


    # 2. Group by 'Candidate' and calculate descriptive statistics for 'Votes'
    print("\nCalculating descriptive statistics for each candidate...")

    # Define the aggregations
    # Pandas' .mode() returns a Series (can have multiple modes).
    # The custom function `calculate_mode` handles this.
    # For standard deviation, pandas default is sample (ddof=1), which is common.
    # For variance, pandas default is also sample (ddof=1).
    descriptive_stats = df.groupby('Candidate')['Votes'].agg(
        Mean_Votes='mean',
        Median_Votes='median',
        Mode_Votes=calculate_mode, # Use our custom mode function
        Std_Dev_Votes='std',
        Variance_Votes='var',
        Min_Votes='min', # Added for extra context
        Max_Votes='max',   # Added for extra context
        Count_Polling_Stations='count' # Number of polling stations they got votes in
    ).reset_index() # To make 'Candidate' a regular column again

    # 3. Display the results
    print("\nDescriptive Statistics per Candidate:")
    print(descriptive_stats)

    # 4. Save the results to a new CSV file
    # Create the 'data' directory if it doesn't exist (similar to histogram script)
    import os
    output_directory = os.path.dirname(OUTPUT_STATS_FILE_PATH)
    if output_directory and not os.path.exists(output_directory): # Check if directory part is not empty
        os.makedirs(output_directory, exist_ok=True)
        print(f"\nCreated directory: {output_directory}")

    try:
        descriptive_stats.to_csv(OUTPUT_STATS_FILE_PATH, index=False, encoding='utf-8-sig')
        print(f"\nDescriptive statistics saved to: {OUTPUT_STATS_FILE_PATH}")
    except Exception as e:
        print(f"\nError saving statistics to CSV: {e}")