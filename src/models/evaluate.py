import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


def main():

    predictions_filapath = "data/prediction.csv"
    metrics_filepath = "metrics/scores.json"
    
    df_X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    df_y_train = pd.read_csv("data/processed_data/y_train.csv")
    df_y_train = df_y_train.astype(float)

    print(df_y_train)

    # Charger le modèle depuis le fichier pickle
    with open('models/gbr_model.pkl', 'rb') as f:
        model = pickle.load(f)

    y_pred = save_model_metrics(model, df_X_train_scaled, df_y_train, metrics_filepath)
    save_predictions_to_csv(y_pred, predictions_filapath)

def save_model_metrics(model, X_train, y_train, file_path):

    # Calcul des métriques
    y_pred = model.predict(X_train)

    metrics = {
        "MSE": mean_squared_error(y_train, y_pred),
        "MAE": mean_absolute_error(y_train, y_pred),
        "R2_Score": r2_score(y_train, y_pred)
    }
   
    # Sauvegarde des métriques dans un fichier JSON
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return y_pred
    
# Fonction pour sauvegarder les prédictions dans un fichier CSV
def save_predictions_to_csv(y_pred, file_path):
    """
    Sauvegarde les prédictions dans un fichier CSV.
    
    :param y_pred: Prédictions du modèle (tableau ou liste).
    :param file_path: Chemin du fichier CSV à sauvegarder.
    """

    # Créer un DataFrame avec les indices et les prédictions
    predictions_df = pd.DataFrame(y_pred)
    
    # Sauvegarder le DataFrame dans un fichier CSV
    predictions_df.to_csv(file_path, index=False)
    print(f"Prédictions sauvegardées dans '{file_path}'")

if __name__ == '__main__':
    main()
