import pandas as pd
import matplotlib.pyplot as plt
import os
import re # For cleaning filenames

# --- Configuration ---
# Assuming your CSV file is named 'election_data.csv' in the same directory as the script
CSV_FILE_PATH = 'data/polska_prezydent2025_tura2_obkw_kandydaci_NATIONAL_FINAL.csv'
OUTPUT_DIR = 'data/histograms/'

# --- Helper function to sanitize filenames ---
def sanitize_filename(name):
    """
    Sanitizes a string to be used as a filename.
    Replaces spaces with underscores and removes characters not typically allowed.
    """
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\._-]', '', name) # Keep word chars, dots, underscores, hyphens
    return name

# --- Main script ---
if __name__ == "__main__":
    # 1. Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")

    # 2. Load the CSV data
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded data from '{CSV_FILE_PATH}'.")
        print(f"DataFrame head:\n{df.head()}")
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        print("Please make sure the CSV file is in the correct location and the path is correct.")
        # Create a dummy CSV for testing if it's missing
        print("\nCreating a dummy 'election_data.csv' for demonstration purposes...")
        dummy_data = """Wojewodztwo_Name,Powiat_MnpP_Name,Gmina_Name,URL_ID,Candidate,Votes
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BARTOSZEWICZ Artur,2
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
        print(f"DataFrame head:\n{df.head()}")


    # 3. Get unique candidate names
    candidates = df['Candidate'].unique()
    print(f"\nFound {len(candidates)} unique candidates: {', '.join(candidates)}")

    # 4. For each candidate, generate and save a histogram
    for candidate_name in candidates:
        print(f"\nProcessing candidate: {candidate_name}")

        # Filter data for the current candidate
        candidate_data = df[df['Candidate'] == candidate_name]

        # Get the 'Votes' for this candidate across different polling stations/URL_IDs
        # Each row for a candidate represents votes in one URL_ID
        votes_per_station = candidate_data['Votes']

        if votes_per_station.empty:
            print(f"  No vote data found for {candidate_name}. Skipping.")
            continue

        # Create histogram
        plt.figure(figsize=(10, 6)) # Create a new figure for each plot
        plt.hist(votes_per_station, bins='auto', color='skyblue', edgecolor='black')
        # 'auto' lets matplotlib decide the optimal number of bins.
        # You can also specify an integer, e.g., bins=20

        plt.title(f'Vote Distribution for {candidate_name}')
        plt.xlabel('Number of Votes per Polling Station (URL_ID)')
        plt.ylabel('Frequency (Number of Polling Stations)')
        plt.grid(axis='y', alpha=0.75)

        # Sanitize candidate name for filename
        safe_filename_candidate = sanitize_filename(candidate_name)
        output_filename = f"{safe_filename_candidate}_histogram.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Save the plot
        try:
            plt.savefig(output_path)
            print(f"  Histogram saved to: {output_path}")
        except Exception as e:
            print(f"  Error saving histogram for {candidate_name}: {e}")

        plt.close() # Close the figure to free up memory and prevent plots from overlapping

    print("\nAll histograms generated and saved.")