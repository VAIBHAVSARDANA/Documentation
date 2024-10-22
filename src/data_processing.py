import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def extract_floor_info(floor_text):
    match = re.match(r'(\w+\s*\w*)\s+out\s+of\s+(\d+)', floor_text)
    if match:
        return match.groups()
    return None, None

def general_cleaning(data):
    # Extract floor information
    data[['floor_number', 'total_floor']] = data['Floor'].apply(extract_floor_info).apply(pd.Series)

    # Convert total_floor to numeric
    data['total_floor'] = pd.to_numeric(data['total_floor'], errors='coerce')

    # Convert floor_number to a numerical representation (if necessary)
    data['floor_number'] = pd.to_numeric(data['floor_number'], errors='coerce')

    # Handle missing values in floor_number and total_floor
    data['floor_number'].fillna(0, inplace=True)
    data['total_floor'].fillna(0, inplace=True)

    # Handle Area Locality column with Label Encoding for saving cleaned data
    encoder = LabelEncoder()
    data['Area Locality Encoded'] = encoder.fit_transform(data['Area Locality'].astype(str))

    # Initialize LabelEncoder for other categorical features
    categorical_columns = ['BHK', 'Furnishing Status', 'City', 'Area Type', 'Tenant Preferred', 'Point of Contact']
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col].astype(str))

    # Handle missing values after conversion
    data.fillna(0, inplace=True)

    # Save cleaned data for Power BI
    data.to_csv('data/cleaned_house_rent_data.csv', index=False)
    print("Cleaned data saved for Power BI.")

    return data

def preprocess_data(data):
    # Perform general cleaning
    data = general_cleaning(data)

    # Drop the original non-numeric columns
    data.drop(columns=['Area Locality', 'Floor'], inplace=True)

    # Date processing: extract Year and Month from 'Posted On'
    data['Posted On'] = pd.to_datetime(data['Posted On'], errors='coerce')  # Handle potential errors
    data['Year Posted'] = data['Posted On'].dt.year
    data['Month Posted'] = data['Posted On'].dt.month
    data.drop(columns=['Posted On'], inplace=True)  # Drop the original date column after extraction

    # Scaling numerical columns
    scaler = StandardScaler()
    numerical_columns = ['Size', 'Rent', 'total_floor', 'floor_number', 'Year Posted', 'Month Posted']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Define target column
    target_column = 'Rent'

    # Split into features (X) and target (y)
    X = data.drop(columns=[target_column])  # Drop 'Rent' (target) and keep only features
    y = data[target_column]
    
    # Save the scaler for future use
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X, y
