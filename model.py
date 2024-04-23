import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("C:/Users/lenovo/Desktop/Desktop Folders/MLProjectFinal/MLProject/credit_rating.csv")

# Drop the S.No. columns
data = data.drop(data.columns[data.columns.str.contains('S.No')], axis=1)

# Handle missing values if any
data.dropna(inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for col in data.columns[data.dtypes == 'object']:
    data[col] = label_encoder.fit_transform(data[col])

# Save feature names
feature_names = data.columns.tolist()

# Splitting the data into features (X) and target variable (y)
X = data.drop('Credit classification', axis=1)
y = data['Credit classification']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Support Vector Machine with cross-validation
svm_classifier = SVC(kernel='rbf', probability=True)
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
grid_search_svm = GridSearchCV(svm_classifier, param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)

# Use best SVM model from GridSearchCV
best_svm_model = grid_search_svm.best_estimator_

# Train Random Forest with cross-validation
rf_classifier = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Use best RF model from GridSearchCV
best_rf_model = grid_search_rf.best_estimator_

# Train Gradient Boosting with cross-validation
gb_classifier = GradientBoostingClassifier(random_state=42)
param_grid_gb = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
grid_search_gb = GridSearchCV(gb_classifier, param_grid_gb, cv=5)
grid_search_gb.fit(X_train, y_train)

# Use best GB model from GridSearchCV
best_gb_model = grid_search_gb.best_estimator_

# Predict probabilities using SVM, RF, and GB models
svm_pred_proba = best_svm_model.predict_proba(X_train)
rf_pred_proba = best_rf_model.predict_proba(X_train)
gb_pred_proba = best_gb_model.predict_proba(X_train)

# Stack SVM, RF, and GB predictions
X_train_stacked = np.hstack((svm_pred_proba, rf_pred_proba, gb_pred_proba))

# Train Logistic Regression on stacked features
lr_classifier_stacked = LogisticRegression()
lr_classifier_stacked.fit(X_train_stacked, y_train)

# Predict probabilities on test set
svm_pred_proba_test = best_svm_model.predict_proba(X_test)
rf_pred_proba_test = best_rf_model.predict_proba(X_test)
gb_pred_proba_test = best_gb_model.predict_proba(X_test)

# Stack SVM, RF, and GB predictions for test set
X_test_stacked = np.hstack((svm_pred_proba_test, rf_pred_proba_test, gb_pred_proba_test))

# Predict using LR model on stacked features
lr_pred = lr_classifier_stacked.predict(X_test_stacked)

# Evaluate LR model
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, average='weighted')
lr_recall = recall_score(y_test, lr_pred, average='weighted')
lr_f1 = f1_score(y_test, lr_pred, average='weighted')

print("Stacked Model Metrics:")
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1 Score:", lr_f1)

# Save the trained model
with open('best_svm_model.pkl', 'wb') as f:
    pickle.dump(best_svm_model, f)

with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)

with open('best_gb_model.pkl', 'wb') as f:
    pickle.dump(best_gb_model, f)

with open('lr_classifier_stacked.pkl', 'wb') as f:
    pickle.dump(lr_classifier_stacked, f)

# Save the scaler object
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)