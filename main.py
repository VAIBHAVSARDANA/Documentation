import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
from src.data_processing import preprocess_data, general_cleaning
from src.utils import save_model, load_model

# Load dataset
data = pd.read_csv('data/House_Rent_Dataset.csv')

# Perform general cleaning and save cleaned data for Power BI
cleaned_data = general_cleaning(data)

# Proceed with further preprocessing for model training
X, y = preprocess_data(cleaned_data)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train new models
def train_models(X_train, y_train):
    print("Training Random Forest...")
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)

    return rf, gb

# Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# Compare new and saved models
def compare_models(new_model, saved_model_path, X_test, y_test):
    new_model_rmse = evaluate_model(new_model, X_test, y_test)

    try:
        saved_model = load_model(saved_model_path)
        saved_model_rmse = evaluate_model(saved_model, X_test, y_test)
        print(f"Saved model RMSE: {saved_model_rmse:.4f}")
        return new_model_rmse, saved_model_rmse
    except FileNotFoundError:
        print("No saved model found. Saving new model.")
        save_model(new_model, saved_model_path)
        return new_model_rmse, float('inf')

# Main function to orchestrate training and saving
def main():
    # Train models
    rf, gb = train_models(X_train, y_train)

    # Compare Random Forest models
    print("\nComparing Random Forest models...")
    rf_new_rmse, rf_saved_rmse = compare_models(rf, 'models/best_model_RandomForest.pkl', X_test, y_test)
    
    if rf_new_rmse < rf_saved_rmse:
        print(f"New Random Forest model is better with RMSE {rf_new_rmse:.4f}. Saving it.")
        save_model(rf, 'models/best_model_RandomForest.pkl')
    else:
        print(f"Old Random Forest model is better with RMSE {rf_saved_rmse:.4f}. Keeping the old model.")
    
    # Compare Gradient Boosting models
    print("\nComparing Gradient Boosting models...")
    gb_new_rmse, gb_saved_rmse = compare_models(gb, 'models/best_model_GradientBoosting.pkl', X_test, y_test)
    
    if gb_new_rmse < gb_saved_rmse:
        print(f"New Gradient Boosting model is better with RMSE {gb_new_rmse:.4f}. Saving it.")
        save_model(gb, 'models/best_model_GradientBoosting.pkl')
    else:
        print(f"Old Gradient Boosting model is better with RMSE {gb_saved_rmse:.4f}. Keeping the old model.")

    print("\nTraining completed. Models compared and best models saved.")

if __name__ == "__main__":
    main()
