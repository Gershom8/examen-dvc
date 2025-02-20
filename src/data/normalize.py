import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def main():

    X_train_filepath = "data/processed_data/X_train.csv"
    X_test_filepath = "data/processed_data/X_test.csv"
    output_filepath = "data/processed_data/"

    process_data(X_train_filepath, X_test_filepath, output_filepath)

def process_data(X_train_filepath, X_test_filepath, output_filepath):
    # Import datasets
    X_train = pd.read_csv(X_train_filepath)
    X_test = pd.read_csv(X_test_filepath)

    #X_train['year'] = pd.to_datetime(X_train['datetime_column']).dt.year
    #X_train['month'] = pd.to_datetime(X_train['datetime_column']).dt.month
    #X_train['day'] = pd.to_datetime(X_train['datetime_column']).dt.day
    #X_train['hour'] = pd.to_datetime(X_train['datetime_column']).dt.hour
    
    #X_test['year'] = pd.to_datetime(X_test['datetime_column']).dt.year
    #X_test['month'] = pd.to_datetime(X_test['datetime_column']).dt.month
    #X_test['day'] = pd.to_datetime(X_test['datetime_column']).dt.day
    #X_test['hour'] = pd.to_datetime(X_test['datetime_column']).dt.hour

    X_train.drop(columns=['date'], inplace=True)
    X_test.drop(columns=['date'], inplace=True)
    
    # Split data into training and testing sets
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    
    # Save dataframes
    save_dataframes(X_train_scaled, X_test_scaled, output_filepath)

def normalize_data(X_train, X_test):
    # Normalize data
    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir en DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled_df, X_test_scaled_df

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    main()
 