import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def main():

    X_train_scaled_filepath = "data/processed_data/X_train_scaled.csv"
    y_train_filepath = "data/processed_data/y_train.csv"

    process_data(X_train_scaled_filepath, y_train_filepath)

def process_data(X_train_scaled_filepath, y_train_filepath):
    # Import datasets
    df_X_train_scaled = pd.read_csv(X_train_scaled_filepath)
    df_y_train = pd.read_csv(y_train_filepath)

    #gridsearch
    trainmodel(df_X_train_scaled, df_y_train)

def trainmodel(X_train_scaled, y_train):

	# Charger le modèle depuis le fichier pickle
	with open('models/best_params.pkl', 'rb') as f:
	    model = pickle.load(f)
	   
	# Entrainement du modèle
	model.fit(X_train_scaled, y_train)

	# Sauvegarder le modèle réentraîné avec pickle
	with open('models/gbr_model.pkl', 'wb') as f:
	    pickle.dump(model, f)
	    
if __name__ == '__main__':
    main()
