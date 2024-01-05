# -*- coding: utf-8 -*-
"""
Name: Cities_API.py
Desc: 
Gather all city names for standarization of the dataset using the API
on https://public.opendatasoft.com/explore/dataset/geonames-postal-code/api/
there's a daily limit of requests to the API so it has been gathered in separate files
'
By HAN Data Cleaning group
"""

import pandas as pd
from os import listdir
import requests
from datetime import datetime
from tqdm import tqdm

def getAdminNames(country_code, admin_type=None):
    base_url = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-postal-code@public/records"

    def fetch_admin_names(admin_level):
        url = f"{base_url}?group_by={admin_level}&limit=-1&refine=country_code%3A%22{country_code}%22"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [result[admin_level] for result in data.get('results', []) if result.get(admin_level)]
        else:
            print(f"Failed to retrieve admin names for {country_code} from {admin_level}: Status code {response.status_code}")
            print(f"Failed URL: {url}")
            print(f"Response: {response.text}")
            return []

    # If admin_type is specified, fetch only that type
    if admin_type:
        return fetch_admin_names(admin_type)

    # Default behavior: try admin_name2 first, then admin_name1
    admin_names = fetch_admin_names('admin_name2')
    if not admin_names:
        print("\nThere are no results for admin_name2, attempting to gather from admin_name1")
        admin_names = fetch_admin_names('admin_name1')

    return admin_names


def getCitiesForAdminName(admin_name, admin_type='admin_name2'):
    base_url = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-postal-code@public/records"
    all_cities = []
    limit = 100  # Adjust as needed
    offset = 0

    while True:
        url = f"{base_url}?limit={limit}&offset={offset}&refine={admin_type}%3A%22{admin_name}%22"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            cities = [record for record in data.get('results', [])]
            all_cities.extend(cities)

            if len(cities) < limit:
                break
            offset += limit
        else:
            print(f"Failed to retrieve cities for {admin_name}: Status code {response.status_code}")
            print(f"Failed URL: {url}")
            print(f"Response: {response.text}")
            break

    # If no results were found for admin_name2, try with admin_name1
    if not all_cities and admin_type == 'admin_name2':
        return getCitiesForAdminName(admin_name, admin_type='admin_name1')

    return all_cities

def main():
    "Main function that runs when the code is executed"
    save_directory = "Cities/"
    print("GATHER CITY DATASET FROM OPENSOFT API")
    print("#"*20)
    print("""in order to gather the data for cities in a country, the country code has to be provided.
          (1) gather country codes from dataset file
          (2) gather country codes from user input
          Write the prefered option's number:""")
    option = input()
    print("\n")
    if option == "1":
        ################### USER HAS TO CHANGE THE FOLLOWING LINE #########################
        directory = "Tables Joined" # Change according to where the directory is
        ###################################################################################
        files = listdir(directory)
        selected_files = {}
        i = 0
        for file in files:
            if ".csv" in file:
                selected_files[i] = file
                print(f"{'('+str(i)+')' :<5} {file}")
                i += 1
        i = int(input("Select the file number you wish to input country codes from: \n"))
        df = pd.read_csv(directory+"/"+selected_files[i])
        print("SELECTED DATASET COLUMN NAMES:\n","\n".join(df.columns))
        country_col = input("Write the column name where the country code is:\n")
        
        country_codes = df[country_col].unique()
        print("selected country codes from the dataset:",country_codes)
    elif option == "2":
        country_input = input("Write the country codes separated by commas:\n")
        country_codes = country_input.split(",")
    
    print("""Choose the admin level for data gathering:
          (1) Use admin_name2 by default and switch to admin_name1 if no results
          (2) Only use admin_name1
          (3) Only use admin_name2
          Write the preferred option's number:""")
    admin_option = input()

    print("Gathering data from API...")
    all_cities_dataset = []

    # Wrap the country_codes iterable with tqdm for a progress bar
    for country_code in tqdm(country_codes, desc="Processing Countries"):
        print(f"Processing {country_code}...")

        # Determine admin_type based on user choice
        if admin_option == '2':
            admin_type = 'admin_name1'
        elif admin_option == '3':
            admin_type = 'admin_name2'
        else:
            admin_type = None  # Default behavior (try admin_name2 first, then admin_name1)

        # Fetch admin names based on the chosen option
        admin_names = getAdminNames(country_code, admin_type)

        for admin_name in tqdm(admin_names, desc=f"Admin Names in {country_code}"):
            cities = getCitiesForAdminName(admin_name, admin_type if admin_type else 'admin_name2')
            all_cities_dataset.extend(cities)
        
        print(f"Finished processing {country_code}.")
    
    # Save the dataframe as a file
    df_api = pd.DataFrame(all_cities_dataset)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename_q = input("What would you like to name the file? (ex. cities_AD) For default name, leave blank\n")
    if filename_q == "":
        # Create filename with timestamp
        filename = f"citiesAPI_{timestamp}.csv"
    else:
        filename = f"{filename_q}.csv"
        
    # Save the DataFrame to CSV
    df_api.to_csv(save_directory+filename, index=False)
    
    print(f"File saved as {filename}")
if __name__ == "__main__":
    main()
