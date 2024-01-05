# -*- coding: utf-8 -*-
"""
name: Standardize Cities:
desc:
With the API data gathered, the cities are standardized using fuzzy similarity and postal codes
"""

from tqdm import tqdm
from fuzzywuzzy import process, fuzz
from datetime import datetime
import pandas as pd
import re
import os

# Function to standardize postal codes (remove special characters and spaces)
def standardize_postal_code(code):
    return re.sub(r'\W+', '', str(code))

# Dictionary to store found matches
found_matches = {}

def find_best_match(row, col_city_name, col_country_code, col_postal_code, cities_df):
    city, country_code, original_postal_code = row[col_city_name], row[col_country_code], row[col_postal_code]

    original_postal_code = str(original_postal_code) if pd.notna(original_postal_code) else ''

    if not isinstance(city, str) or re.match(r'^[\W_]+$', city) or not city.strip():
        return pd.Series([None, None])

    if (city, country_code) in found_matches:
        return pd.Series(found_matches[(city, country_code)])

    filtered_cities = cities_df[cities_df['country_code'] == country_code]

    if filtered_cities.empty:
        found_matches[(city, country_code)] = (None, None)
        return pd.Series([None, None])

    # Attempt to find an exact match for postal code
    exact_match = filtered_cities[filtered_cities['Standardized_Postal_Code'] == original_postal_code]
    if not exact_match.empty:
        matched_city = exact_match.iloc[0]['place_name']
        matched_postal_code = exact_match.iloc[0]['postal_code']
        found_matches[(city, country_code)] = (matched_city, matched_postal_code)
        return pd.Series([matched_city, matched_postal_code])

    # Preparing choices for fuzzy matching
    city_choices = filtered_cities['place_name'].str.upper().unique().tolist()
    city_choices = [str(c) for c in city_choices if isinstance(c, str)]

    # Fuzzy matching for the city
    best_city = process.extractOne(city.upper(), city_choices, scorer=fuzz.partial_ratio)

    # If a city match is found, filter the postal codes for this city
    if best_city:
        potential_postal_codes = filtered_cities[filtered_cities['place_name'].str.upper() == best_city[0]]['postal_code'].unique().tolist()
        potential_postal_codes = [str(p) for p in potential_postal_codes if pd.notna(p)]
        # Fuzzy matching for the postal code
        best_postal_code = process.extractOne(original_postal_code, potential_postal_codes, scorer=fuzz.ratio) if original_postal_code else (None, None)
    else:
        best_postal_code = (None, None)

    best_city_name = best_city[0] if best_city else None
    best_postal_code_value = best_postal_code[0] if best_postal_code else None

    found_matches[(city, country_code)] = (best_city_name, best_postal_code_value)
    return pd.Series([best_city_name, best_postal_code_value])

def main():
    ################### USER HAS TO CHANGE THE FOLLOWING LINE #########################
    API_cities_dir= "Cities/" # change according to where the "All_Gathered_Cities_API.csv" is
    ###################################################################################
   
    cities_df = pd.read_csv(API_cities_dir+"All_Gathered_Cities_API.csv")
    
    ################### USER HAS TO CHANGE THE FOLLOWING LINE #########################
    directory = "Tables Joined" # Change according to where the directory of datasets
    ###################################################################################
    files = os.listdir(directory)
    selected_files = {}
    i = 0
    for file in files:
        if ".csv" in file:
            selected_files[i] = file
            print(f"{'('+str(i)+')' :<5} {file}")
            i += 1
    i = int(input())
    df = pd.read_csv(directory+"/"+selected_files[i])
    
    print("SELECTED DATASET COLUMN NAMES:")
    columns = {}
    for c in df.columns:
        print(c)
    print("\nWrite the name of the column containing the Postal Code:")
    col_postal_code = input()
    print("\nWrite the name of the column containing the Country Code:")
    col_country_code = input()
    print("\nWrite the name of the column containing the city name")
    col_city_name = input()
    
    # Standardize postal codes in both dataframes
    df['Standardized_DCPC'] = df[col_postal_code].apply(standardize_postal_code)
    cities_df['Standardized_Postal_Code'] = cities_df['postal_code'].apply(standardize_postal_code)

    # Apply the function to the DataFrame with tqdm for progress tracking
    tqdm.pandas(desc="Finding Matches")
    df[['Standardized_City', 'Similar_Postal_Code']] = df.progress_apply(lambda row: find_best_match(row, col_city_name, col_country_code, 'Standardized_DCPC', cities_df), axis=1)
    
    # Save the final file
    ################### USER HAS TO CHANGE THE FOLLOWING LINE #########################
    save_directory = "Cities/"
    ###################################################################################
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename_q = input("What would you like to name the file? (ex. cities_AD) For default name, leave blank\n")
    if filename_q == "":
        # Create filename with timestamp
        filename = f"{selected_files[i][:-4]}_{timestamp}.csv"
    else:
        filename = f"{filename_q}.csv"
        
    # Save the DataFrame to CSV
    df.to_csv(save_directory+filename, index=False)
if __name__ == "__main__":
    main()
