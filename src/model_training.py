import os
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np
from .utils import save_model, load_model

def train_and_compare_models(X_train, X_test, y_train, y_test, new_models):
    best_model = None
    best_rmse = float('inf')

    for model_name, model in new_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'{model_name} RMSE: {rmse}')

        # Load the previous best model (if exists) and compare
        model_path = f'models/best_model_{model_name}.pkl'
        if os.path.exists(model_path):
            old_model = load_model(model_path)
            old_rmse = np.sqrt(mean_squared_error(y_test, old_model.predict(X_test)))

            if rmse < old_rmse:
                print(f'New {model_name} model is better with RMSE {rmse}. Saving it.')
                save_model(model, model_name)
            else:
                print(f'Old {model_name} model is better with RMSE {old_rmse}.')
        else:
            print(f'No previous {model_name} model found. Saving the new model.')
            save_model(model, model_name)

        # Keep track of the best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    return best_model
