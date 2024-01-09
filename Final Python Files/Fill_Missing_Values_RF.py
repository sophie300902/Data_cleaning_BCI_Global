import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats

def load_data(directory):
    """Load CSV files from a specified directory into a dictionary of dataframes."""
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    return {file: pd.read_csv(os.path.join(directory, file)) for file in files}

def summarize_columns(data):
    """Print a summary of the dataset columns, including type, unique values, missing values, and sample values."""
    print("Summary of Dataset Columns:")
    print(f"{'Column':<20} {'Type':<10} {'Unique':<10} {'Missing':<10} {'Samples':<30}")
    for column in data.columns:
        col_type = data[column].dtype
        unique_count = data[column].nunique()
        missing_count = str(round(data[column].isnull().sum()/len(data)*100)) + "%"
        samples = data[column].dropna().unique()[:3]
        print(f"{column:<20} {str(col_type):<10} {unique_count:<10} {missing_count:<10} {str(samples):<30}")
        
def analyze_dataframe_for_recommendations(data, unique_threshold=1000, missing_threshold=0.8):
    """Analyze the dataframe and suggest key column candidates and columns with high missing values."""
    total_rows = len(data)
    key_column_candidates = []
    high_missing_columns = []

    for column in data.columns:
        uniques = data[column].nunique()
        missing_ratio = data[column].isnull().sum() / total_rows

        if uniques > unique_threshold:
            key_column_candidates.append(column)
        if missing_ratio > missing_threshold:
            high_missing_columns.append(column)

    return key_column_candidates, high_missing_columns

def convert_to_numeric(data):
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='ignore')
    return data

def apply_label_encoding(data):
    encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            encoder = LabelEncoder()
            data[column] = data[column].astype(str)
            data[column] = encoder.fit_transform(data[column])
            encoders[column] = encoder
    return data, encoders

def one_hot_encode(data, return_dummy_columns_mapping=False):
    dummy_columns_mapping = {}
    for column in data.select_dtypes(include=['object']).columns:
        dummies = pd.get_dummies(data[column], prefix=column, drop_first=True, dummy_na=True)
        dummy_columns_mapping[column] = dummies.columns
        data = pd.concat([data.drop(column, axis=1), dummies], axis=1)

    if return_dummy_columns_mapping:
        return data, dummy_columns_mapping
    else:
        return data


def impute_with_random_forest(data, column, key_columns=[]):
    # Separate the column to be imputed
    target = data[column]
    features = data.drop(columns=key_columns + [column])

    # Apply one-hot encoding
    features_encoded = one_hot_encode(features)

    # Fill missing values in features with a placeholder
    features_encoded = features_encoded.fillna(-9999)

    # Determine if the target column is categorical or numeric
    is_numeric = pd.api.types.is_numeric_dtype(target)

    # Split data into sets with known and unknown target values
    known = features_encoded[~target.isna()]
    unknown = features_encoded[target.isna()]
    known_target = target[~target.isna()]

    # Convert features to NumPy array to avoid UserWarning
    known_np = known.to_numpy()
    unknown_np = unknown.to_numpy()
    known_target_np = known_target.to_numpy()

    # Select appropriate Random Forest model
    if is_numeric:
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    # Fit the model using NumPy array
    model.fit(known_np, known_target_np)

    # Predict using NumPy array
    predicted = model.predict(unknown_np)

    # For categorical data, calculate certainty
    if not is_numeric:
        preds = np.array([tree.predict(unknown_np) for tree in model.estimators_])
        mode_preds, _ = stats.mode(preds, axis=0, keepdims=True)
        certainty = np.sum(preds == mode_preds, axis=0) / preds.shape[0]
        certainty_column_name = f"{column}_certainty_rf"
        data.loc[data[column].isna(), certainty_column_name] = certainty

    # Create a new column combining original and imputed values
    imputed_column_name = f"{column}_imputed_rf"
    # Fill the new column with original values
    data[imputed_column_name] = data[column]
    # Update only the missing values with the predictions
    data.loc[data[column].isna(), imputed_column_name] = predicted

    return data
    
def select_dataframe(dataframes):
    """Let the user select a dataframe from a list and return both the dataframe and its filename."""
    while True:
        print("SELECT A DATAFRAME:\n")
        files_csv = list(dataframes.keys())
        for i, file in enumerate(files_csv):
            print(f"{i:<5} {file}")

        df_index = input("Select a dataframe number (or 'exit' to quit):\n").lower()
        if df_index == 'exit':
            return None, None

        try:
            df_index = int(df_index)
            return dataframes[files_csv[df_index]], files_csv[df_index]
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")

def main():
    """Main function to run the data imputation script."""
    directory = "Tables Joined"
    imputed_directory = "Imputed Data"
    dataframes = load_data(directory)

    selected_df, original_filename = select_dataframe(dataframes)
    if selected_df is None:
        return

    summarize_columns(selected_df)
    key_col_candidates, high_missing_cols = analyze_dataframe_for_recommendations(selected_df)
    print(f"Recommended key columns: {','.join(key_col_candidates)}")
    print(f"Columns with >80% missing values: {','.join(high_missing_cols)}")

    # Ask user for key columns with recommendations in mind
    key_columns_input = input("Enter key column names separated by commas (if any, otherwise leave blank):\n")
    key_columns = [col.strip() for col in key_columns_input.split(',')] if key_columns_input else []

    # Remove duplicate rows
    initial_row_count = selected_df.shape[0]
    selected_df = selected_df.drop_duplicates()
    print(f"Removed {initial_row_count - selected_df.shape[0]} duplicate rows.")

    impute_choice = input("Would you like to impute all columns or a single column? Enter 'all' or 'single':\n").lower()
    columns_to_impute = selected_df.columns.difference(key_columns)

    if impute_choice == 'single':
        column = input("Select a column for imputation:\n")
        if column not in columns_to_impute:
            print("Column not found or is a key column. Please try again.")
            return
        columns_to_impute = [column]

    for column in columns_to_impute:
        if selected_df[column].isnull().any():
        
            # Random Forest Imputation
            print(f"Imputing missing values in '{column}' using Random Forest...")
            selected_df = impute_with_random_forest(selected_df, column, key_columns=key_columns)

    # Save the imputed dataframe outside the loop
    if not os.path.exists(imputed_directory):
        os.makedirs(imputed_directory)

    imputed_filename = f"{os.path.splitext(original_filename)[0]}_imputed.csv"
    imputed_filepath = os.path.join(imputed_directory, imputed_filename)
    selected_df.to_csv(imputed_filepath, index=False)
    print(f"Imputed data saved to '{imputed_filepath}'")

if __name__ == "__main__":
    main()

