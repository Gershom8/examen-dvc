import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

def main():

    input_filepath = "data/raw_data/raw.csv"
    output_filepath = "data/processed_data/"

    process_data(input_filepath, output_filepath)

def process_data(input_filepath, output_filepath):
    # Import datasets
    df = pd.read_csv(input_filepath)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Save dataframes
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    main()
 