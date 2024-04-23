from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained models and scaler
with open('best_svm_model.pkl','rb') as f:
    best_svm_model = pickle.load(f)

with open('best_rf_model.pkl','rb') as f:
    best_rf_model = pickle.load(f)

with open('best_gb_model.pkl','rb') as f:
    best_gb_model = pickle.load(f)

with open('lr_classifier_stacked.pkl','rb') as f:
    lr_classifier_stacked = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Mapping dictionaries for label encoding
encoding_mappings = {
    'CHK_ACCT': {'0DM': 0, 'less-200DM': 1, 'no-account': 2, 'over-200DM': 3},
    'History': {'all-paid-duly': 0, 'bank-paid-duly': 1, 'critical': 2, 'delay': 3, 'duly-till-now': 4},
    'Purpose of credit': {'business': 0, 'domestic-app': 1, 'education': 2, 'furniture': 3, 'new-car': 4,
                          'others': 5, 'radio-tv': 6, 'repairs': 7, 'retraining': 8, 'used-car': 9},
    'Balance in Savings A/C': {'less1000DM': 0, 'less100DM': 1, 'less500DM': 3, 'over1000DM': 4, 'unknown': 5},
    'Employment': {'four-years': 0, 'one-year': 1, 'over-seven': 2, 'seven-years': 3, 'unemployed': 4},
    'Marital status': {'female-divorced': 0, 'male-divorced': 1, 'married-male': 2, 'single-male': 3},
    'Co-applicant': {'co-applicant': 0, 'guarantor': 1, 'none': 2},
    'Real Estate': {'building-society': 0, 'car': 1, 'none': 2, 'real-estate': 3},
    'Other installment': {'bank': 0, 'none': 1, 'stores': 2},
    'Residence': {'free': 0, 'own': 1, 'rent': 2},
    'Job': {'management': 0, 'skilled': 1, 'unemployed-non-resident': 2, 'unskilled-resident': 3},
    'Phone': {'no': 0, 'yes': 1},
    'Foreign': {'no': 0, 'yes': 1}
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    features = {
        'CHK_ACCT': request.form['CHK_ACCT'],
        'Duration': int(request.form['Duration']),
        'History': request.form['History'],
        'Purpose of credit': request.form['Purpose_of_credit'],
        'Credit Amount': int(request.form['Credit_Amount']),
        'Balance in Savings A/C': request.form['Balance_in_Savings_Acct'],
        'Employment': request.form['Employment'],
        'Install_rate': int(request.form['Install_rate']),
        'Marital status': request.form['Marital_status'],
        'Co-applicant': request.form['Co_applicant'],
        'Present Resident': int(request.form['Present_Resident']),
        'Real Estate': request.form['Real_Estate'],
        'Age': int(request.form['Age']),
        'Other installment': request.form['Other_installment'],
        'Residence': request.form['Residence'],
        'Num_Credits': int(request.form['Num_Credits']),
        'Job': request.form['Job'],
        'No. dependents': int(request.form['No_dependents']),
        'Phone': request.form['Phone'],
        'Foreign': request.form['Foreign']
    }

    # Apply label encoding to the features
    for feature, mapping in encoding_mappings.items():
        features[feature] = mapping[features[feature]]

    # Convert features to DataFrame
    input_df = pd.DataFrame([features])

    # Scale the input data
    input_features_scaled = scaler.transform(input_df)

    # Predict probabilities using SVM, RF, and GB models
    svm_pred_proba = best_svm_model.predict_proba(input_features_scaled)
    rf_pred_proba = best_rf_model.predict_proba(input_features_scaled)
    gb_pred_proba = best_gb_model.predict_proba(input_features_scaled)

    # Stack SVM, RF, and GB predictions
    input_features_stacked = np.hstack((svm_pred_proba, rf_pred_proba, gb_pred_proba))

    # Make prediction using the Logistic Regression model
    pred = lr_classifier_stacked.predict(input_features_stacked)

    # Return prediction label
    prediction_label = 'good' if pred[0] == 1 else 'bad'

    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
