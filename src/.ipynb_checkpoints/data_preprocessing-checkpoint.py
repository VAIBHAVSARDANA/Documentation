import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def preprocess_data(data):
    # Encoding categorical features
    encoder = LabelEncoder()
    data['BHK'] = encoder.fit_transform(data['BHK'])
    data['Furnishing Status'] = encoder.fit_transform(data['Furnishing Status'])
    data['City'] = encoder.fit_transform(data['City'])
    data['Area Type'] = encoder.fit_transform(data['Area Type'])
    data['Tenant Preferred'] = encoder.fit_transform(data['Tenant Preferred'])
    data['Point of Contact'] = encoder.fit_transform(data['Point of Contact'])

    # Date processing
    data['Posted On'] = pd.to_datetime(data['Posted On'])
    data['Year Posted'] = data['Posted On'].dt.year
    data['Month Posted'] = data['Posted On'].dt.month

    # Scaling numerical columns
    scaler = StandardScaler()
    data[['Size', 'Rent']] = scaler.fit_transform(data[['Size', 'Rent']])
    
    # Save the scaler for future use
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return data
