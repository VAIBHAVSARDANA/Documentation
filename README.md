# Documentation
Rent Price Prediction
This project aims to predict rent prices using machine learning models based on various features such as BHK, city, furnishing status, size, and more. The goal is to train different machine learning models, compare their performance, and keep the most efficient model (based on RMSE) for future use.

Project Structure
bash
Copy code
rent-price-prediction/
│
├── data/
│   └── House_Rent_Dataset.csv       # Raw dataset containing rent data
│
├── models/                          # Directory for saving the trained models
│   ├── best_model_RandomForest.pkl  # Best performing Random Forest model (saved)
│   ├── best_model_GradientBoosting.pkl # Best performing Gradient Boosting model (saved)
│   └── scaler.pkl                   # Scaler used for data normalization
│
├── src/                             # Source code folder
│   ├── __init__.py                  # Marks this as a Python package
│   ├── data_processing.py           # Data preprocessing functions (e.g., encoding, scaling)
│   ├── model_training.py            # Functions for training and comparing models
│   ├── utils.py                     # Helper functions for saving/loading models
│
├── main.py                          # Script to train models and compare them
├── README.md                        # This readme file
└── requirements.txt                 # Python dependencies for the project
Prerequisites
Before running this project, you need to have Python installed along with the necessary dependencies. The dependencies can be installed using the requirements.txt file.

Installing Dependencies
To install the required dependencies, run the following command:

bash
Copy code
pip install -r requirements.txt
Dataset
The dataset is stored in the data/House_Rent_Dataset.csv file. It contains various features such as:

BHK (number of bedrooms)
Size (size in square feet)
Furnishing Status (Furnished, Semi-Furnished, etc.)
City (Location of the property)
Rent (Target variable)
Area Type
Tenant Preferred
Bathroom
And more.
How It Works
Data Preprocessing
The data_processing.py module handles preprocessing tasks like:

Label encoding of categorical features
Standard scaling of numerical features (like Size and Rent)
Parsing date columns for extracting the year and month
Model Training and Comparison
Models Trained: Random Forest, Gradient Boosting
Comparison Metric: Root Mean Square Error (RMSE)
The model_training.py file contains functions to train new models, compare their RMSE with previously trained models, and keep the most efficient one.

Trains new models (Random Forest and Gradient Boosting).
Compares their performance with already saved models (if they exist).
Saves the best-performing model to the models/ directory.
Helper Functions
The utils.py file provides utility functions to:

Save models to the models/ directory.
Load previously saved models for comparison.
How to Run
Preprocess the Data:

The preprocessing is automatically handled when you run the training script. Ensure the dataset is in data/House_Rent_Dataset.csv.

Train and Compare Models:

You can run the main.py file to train and compare models:

bash
Copy code
python main.py
This will:

Train new Random Forest and Gradient Boosting models.
Compare them with previously saved models (if available).
Save the better-performing model.
Evaluate Performance:

After running the script, it will print out the RMSE of each model and indicate whether the new model is better than the saved one.

Example Output
vbnet
Copy code
RandomForest RMSE: 0.2345
New RandomForest model is better with RMSE 0.2345. Saving it.
GradientBoosting RMSE: 0.2567
Old GradientBoosting model is better with RMSE 0.2456.
Future Work
Add additional models for comparison (e.g., XGBoost, LightGBM).
Implement cross-validation for more robust model performance evaluation.
Automate feature selection and hyperparameter tuning.
Explore different metrics such as Mean Absolute Error (MAE).
Requirements
Python 3.x
pandas
numpy
scikit-learn
License
This project is licensed under the MIT License - see the LICENSE file for details.