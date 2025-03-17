import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

np.random.seed(42)
size = 10000  # Number of transactions

# Generate synthetic data
df = pd.DataFrame({
    "Transaction_Amount": np.random.randint(10, 5000, size),
    "Transaction_Type": np.random.choice(["Online", "In-Person"], size),
    "Location": np.random.choice(["Domestic", "International"], size),
    "Time_of_Day": np.random.choice(["Morning", "Afternoon", "Night"], size),
    "Card_Type": np.random.choice(["Debit", "Credit"], size),
    "Previous_Fraud_History": np.random.choice([0, 1], size),
    "Account_Age": np.random.randint(1, 10, size),
    "Credit_Utilization": np.random.uniform(0.1, 1.0, size),
    "Fraudulent": np.random.choice([0, 1], size, p=[0.98, 0.02])  # 98% Non-Fraud, 2% Fraud
})

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target (y)
X = df.drop(columns=["Fraudulent"])
y = df["Fraudulent"]

# Split dataset into training and testing sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for oversampling
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Function to get user input
def get_user_input():
    print("Please enter the following transaction details:")

    # Collect user input for each feature
    transaction_data = {}
    transaction_data['Transaction_Amount'] = float(input("Transaction Amount: "))
    transaction_data['Transaction_Type'] = input("Transaction Type (Online/In-Person): ")
    transaction_data['Location'] = input("Location (Domestic/International): ")
    transaction_data['Time_of_Day'] = input("Time of Day (Morning/Afternoon/Night): ")
    transaction_data['Card_Type'] = input("Card Type (Debit/Credit): ")
    transaction_data['Previous_Fraud_History'] = int(input("Previous Fraud History (0/1): "))
    transaction_data['Account_Age'] = int(input("Account Age (in years): "))
    transaction_data['Credit_Utilization'] = float(input("Credit Utilization (0.1 to 1.0): "))

    return transaction_data

# Function to predict fraud based on user input
def predict_fraud(user_input):
    # Convert input to DataFrame
    customer_df = pd.DataFrame([user_input])

    # Encoding categorical variables using label encoding (match training data encoding)
    label_encoders = {}
    for column in customer_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        customer_df[column] = le.fit_transform(customer_df[column])
        label_encoders[column] = le  # Store encoder for later use

    # Perform one-hot encoding to match the training data columns
    customer_df = pd.get_dummies(customer_df, drop_first=True)
    
    # Align the input data with the model's training data columns (ensure all columns are present)
    missing_cols = set(X.columns) - set(customer_df.columns)
    for c in missing_cols:
        customer_df[c] = 0
    customer_df = customer_df[X.columns]  # Reorder columns to match the training data
    
    # Scale the numerical columns using StandardScaler (use same scaler as used during training)
    num_cols = ['Transaction_Amount', 'Account_Age', 'Credit_Utilization']
    customer_df[num_cols] = scaler.transform(customer_df[num_cols])

    # Predict the fraud status
    prediction = model.predict(customer_df)
    
    return prediction[0]

# Get user input
user_data = get_user_input()

# Predict whether the transaction is fraudulent or not
prediction = predict_fraud(user_data)

# Output the prediction result
if prediction == 1:
    print("\nThis transaction is a credit card fraud.")
else:
    print("\nThis transaction is not a credit card fraud.")
