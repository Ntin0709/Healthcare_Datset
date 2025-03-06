# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# Load saved artifacts
def load_artifacts(output_dir='./model_outputs'):
    model = joblib.load(f"{output_dir}/random_forest_model.pkl")
    scaler = joblib.load(f"{output_dir}/scaler.pkl")
    with open(f"{output_dir}/selected_features.json", 'r') as f:
        selected_features = json.load(f)
    with open(f"{output_dir}/target_encodings.json", 'r') as f:
        encodings = json.load(f)
    return model, scaler, selected_features, encodings

# Preprocess new data
def preprocess_new_data(new_data, scaler, selected_features, encodings):
    new_data['date_of_admission'] = pd.to_datetime(new_data['date_of_admission'])
    
    le = LabelEncoder()
    categorical_cols = ['gender', 'blood_type', 'medical_condition', 'admission_type', 
                        'insurance_provider', 'medication', 'test_results']
    for col in categorical_cols:
        if col in new_data.columns:
            new_data[col] = le.fit_transform(new_data[col])
    
    # Engineer all features from training
    new_data['severity_proxy'] = new_data['admission_type'] * new_data['medical_condition']
    new_data['billing_log'] = np.log1p(new_data['billing_amount'])
    new_data['age_squared'] = new_data['age'] ** 2
    new_data['admission_month'] = new_data['date_of_admission'].dt.month
    new_data['admission_year'] = new_data['date_of_admission'].dt.year
    new_data['insurance_target'] = new_data['insurance_provider'].map(encodings['insurance_target'])
    new_data['medication_target'] = new_data['medication'].map(encodings['medication_target'])
    new_data['test_results_target'] = new_data['test_results'].map(encodings['test_results_target'])
    new_data['avg_stay_condition'] = new_data['medical_condition'].map(encodings['avg_stay_condition'])
    
    X_new = new_data.drop(['name', 'doctor', 'hospital', 'date_of_admission', 'discharge_date', 
                           'room_number'], axis=1, errors='ignore')
    
    print("Features in preprocessed data:", list(X_new.columns))
    
    missing_features = [f for f in selected_features if f not in X_new.columns]
    if missing_features:
        raise KeyError(f"Missing features in new data: {missing_features}")
    
    X_new = X_new[selected_features]  # Keep only and order by selected_features
    
    numerical_cols = ['age', 'billing_log', 'age_squared', 'avg_stay_condition', 
                      'insurance_target', 'medication_target', 'test_results_target']
    available_cols = [col for col in numerical_cols if col in X_new.columns]
    X_new[available_cols] = scaler.transform(X_new[available_cols])
    
    return X_new

# Inference function
def predict_length_of_stay(new_data, model, scaler, selected_features, encodings):
    X_new = preprocess_new_data(new_data, scaler, selected_features, encodings)
    pred_log = model.predict(X_new)
    pred = np.expm1(pred_log)
    return pred

# Main execution
if __name__ == "__main__":
    model, scaler, selected_features, encodings = load_artifacts()
    print("Loaded model, scaler, and features:", selected_features)
    
    new_data = pd.DataFrame({
        'name': ['John Doe'],
        'age': [45],
        'gender': ['Male'],
        'blood_type': ['A+'],
        'medical_condition': ['Diabetes'],
        'date_of_admission': ['2025-01-01'],
        'discharge_date': [np.nan],
        'doctor': ['Dr. Smith'],
        'hospital': ['City Hospital'],
        'insurance_provider': ['Medicare'],
        'billing_amount': [5000],
        'room_number': [101],
        'admission_type': ['Emergency'],
        'medication': ['Insulin'],
        'test_results': ['Abnormal']
    })
    
    predictions = predict_length_of_stay(new_data, model, scaler, selected_features, encodings)
    new_data['predicted_length_of_stay'] = predictions
    print("\nPredictions:")
    print(new_data[['name', 'predicted_length_of_stay']])
    
    new_data.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to 'predictions.csv'")