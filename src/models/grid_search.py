import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
import pickle


def main():

    X_train_scaled_filepath = "data/processed_data/X_train_scaled.csv"
    y_train_filepath = "data/processed_data/y_train.csv"

    process_data(X_train_scaled_filepath, y_train_filepath)

def process_data(X_train_scaled_filepath, y_train_filepath):
    # Import datasets
    df_X_train_scaled = pd.read_csv(X_train_scaled_filepath)
    df_y_train = pd.read_csv(y_train_filepath)

    #gridsearch
    gridsearch(df_X_train_scaled, df_y_train)

def gridsearch(X_train_scaled, y_train):
  
    # Définir le modèle de régression logistique
    knn = KNeighborsRegressor()

    # Définir la grille des hyperparamètres à tester
    param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],  # Nombre de voisins
            'weights': ['uniform', 'distance'],  # Pondération des voisins
            'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance utilisée
    }

    # Appliquer GridSearchCV
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Afficher les meilleurs hyperparamètres
    #print("Meilleurs hyperparamètres :", grid_search.best_params_)

    # Le meilleur modèle
    best_log_reg = grid_search.best_estimator_
   
    # Sauvegarder le modèle entraîné dans un fichier .pkl
    with open('models/best_params.pkl', 'wb') as f:
        pickle.dump(best_log_reg, f)
        #print("Modèle sauvegardé dans 'models/best_log_reg_model.pkl'")

if __name__ == '__main__':
    main()
 

