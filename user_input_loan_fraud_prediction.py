import pandas as pd
import numpy as np
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Initialize the Faker object for generating dummy data
fake = Faker()
Faker.seed(42)
random.seed(42)

# Define possible values for columns
loan_types = ['Personal Loan', 'Home Loan', 'Car Loan', 'Education Loan', 'Business Loan']
loan_purposes = {
    'Personal Loan': ['Medical Emergency', 'Vacation', 'Wedding', 'Debt Consolidation'],
    'Home Loan': ['New Home Purchase', 'Home Renovation', 'Mortgage Refinance'],
    'Car Loan': ['New Car Purchase', 'Used Car Purchase'],
    'Education Loan': ['College Tuition', 'Study Abroad', 'Skill Development Course'],
    'Business Loan': ['Startup Funding', 'Business Expansion', 'Working Capital']
}
employment_types = ['Salaried', 'Self-Employed']
industries = ['IT', 'Healthcare', 'Retail', 'Manufacturing', 'Education']
marital_status_options = ['Single', 'Married', 'Divorced', 'Widowed']
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
payment_behaviors = ['On-time Payments', 'Missed Payments', 'Late Payments']
income_stability = ['Stable', 'Unstable']
fraud_labels = [0, 1]  # 0 = Legitimate, 1 = Fraud

# Generate some random data (1000 entries)
data = []
for _ in range(1000):
    loan_type = random.choice(loan_types)
    row = {
        'Name': fake.name(),
        'Age': random.randint(21, 65),
        'Gender': random.choice(['Male', 'Female']),
        'Marital Status': random.choice(marital_status_options),
        'Employment Type': random.choice(employment_types),
        'Industry': random.choice(industries),
        'Years with Current Employer': random.randint(0, 20),
        'Education Level': random.choice(education_levels),
        'CIBIL Score': random.randint(300, 900),
        'Loan Type': loan_type,
        'Loan Purpose': random.choice(loan_purposes[loan_type]),
        'Loan Amount Requested': random.randint(50000, 5000000),
        'Loan Tenure': random.choice([12, 24, 36, 60, 120, 180, 240]),  # Months
        'Existing Loans': random.randint(0, 5),
        'Loan Repayment History': random.choice(payment_behaviors),
        'DTI Ratio': round(random.uniform(0.1, 1.0), 2),
        'Credit Utilization Ratio': round(random.uniform(0.1, 0.9), 2),
        'Credit Card Payment Behavior': random.choice(payment_behaviors),
        'Monthly Income': random.randint(30000, 200000),
        'Income Stability': random.choice(income_stability),
        'Total Savings & Deposits': random.randint(10000, 500000),
        'Bank Account Activity': random.choice(['Active', 'Dormant']),
        'Large Transactions Before Loan': random.choice(['Yes', 'No']),
        'Number of Credit Cards': random.randint(1, 10),
        'Total Credit Limit': random.randint(50000, 1000000),
        'Utility & Rent Payments History': random.choice(payment_behaviors),
        'Fraud': random.choices(fraud_labels, weights=[90, 10])[0]  # ~10% Fraud cases
    }
    data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Apply label encoding to categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoder for later use

# Define numeric columns to scale
scaler = StandardScaler()
num_cols = ['Age', 'CIBIL Score', 'Loan Amount Requested', 'Loan Tenure',
            'Existing Loans', 'DTI Ratio', 'Credit Utilization Ratio',
            'Monthly Income', 'Total Savings & Deposits',
            'Number of Credit Cards', 'Total Credit Limit']

# Scale the numeric columns
df[num_cols] = scaler.fit_transform(df[num_cols])

# Define features (X) and target (y)
X = df.drop(columns=['Fraud'])
y = df['Fraud']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

# Train the model
model.fit(X_train, y_train)

# Prediction function
def predict_fraud(customer_data):
    # Convert input to DataFrame (for consistency with model training)
    customer_df = pd.DataFrame([customer_data])

    # Apply the same preprocessing steps for categorical columns
    for column in customer_df.select_dtypes(include=['object']).columns:
        # Only apply label encoding to columns that exist in the label encoders dictionary
        if column in label_encoders:
            customer_df[column] = customer_df[column].apply(lambda x: x if x is None else label_encoders[column].transform([x])[0] if x is not None else None)
        else:
            customer_df[column] = customer_df[column].fillna(-1)  # Fill missing categorical values with a placeholder if needed

    # Apply scaling to numerical columns all at once
    customer_df[num_cols] = scaler.transform(customer_df[num_cols])

    # Make the prediction
    prediction = model.predict(customer_df)
    
    return prediction[0]

# Input function
def get_user_input():
    print("Please enter the following information (type 'NA' if not available):")
    
    customer_data = {}
    customer_data['Name'] = input("Name: ")
    customer_data['Age'] = input("Age (leave blank for NA): ")
    customer_data['Gender'] = input("Gender (Male/Female, leave blank for NA): ")
    customer_data['Marital Status'] = input("Marital Status (Single/Married/Divorced/Widowed, leave blank for NA): ")
    customer_data['Employment Type'] = input("Employment Type (Salaried/Self-Employed, leave blank for NA): ")
    customer_data['Industry'] = input("Industry (IT/Healthcare/Retail/Manufacturing/Education, leave blank for NA): ")
    customer_data['Years with Current Employer'] = input("Years with Current Employer (leave blank for NA): ")
    customer_data['Education Level'] = input("Education Level (High School/Bachelor/Master/PhD, leave blank for NA): ")
    customer_data['CIBIL Score'] = input("CIBIL Score (leave blank for NA): ")
    customer_data['Loan Type'] = input("Loan Type (Personal Loan/Home Loan/Car Loan/Education Loan/Business Loan, leave blank for NA): ")
    customer_data['Loan Purpose'] = input("Loan Purpose (leave blank for NA): ")
    customer_data['Loan Amount Requested'] = input("Loan Amount Requested (leave blank for NA): ")
    customer_data['Loan Tenure'] = input("Loan Tenure (in months, leave blank for NA): ")
    customer_data['Existing Loans'] = input("Existing Loans (leave blank for NA): ")
    customer_data['Loan Repayment History'] = input("Loan Repayment History (On-time Payments/Missed Payments/Late Payments, leave blank for NA): ")
    customer_data['DTI Ratio'] = input("DTI Ratio (leave blank for NA): ")
    customer_data['Credit Utilization Ratio'] = input("Credit Utilization Ratio (leave blank for NA): ")
    customer_data['Credit Card Payment Behavior'] = input("Credit Card Payment Behavior (On-time Payments/Missed Payments/Late Payments, leave blank for NA): ")
    customer_data['Monthly Income'] = input("Monthly Income (leave blank for NA): ")
    customer_data['Income Stability'] = input("Income Stability (Stable/Unstable, leave blank for NA): ")
    customer_data['Total Savings & Deposits'] = input("Total Savings & Deposits (leave blank for NA): ")
    customer_data['Bank Account Activity'] = input("Bank Account Activity (Active/Dormant, leave blank for NA): ")
    customer_data['Large Transactions Before Loan'] = input("Large Transactions Before Loan (Yes/No, leave blank for NA): ")
    customer_data['Number of Credit Cards'] = input("Number of Credit Cards (leave blank for NA): ")
    customer_data['Total Credit Limit'] = input("Total Credit Limit (leave blank for NA): ")
    customer_data['Utility & Rent Payments History'] = input("Utility & Rent Payments History (On-time Payments/Missed Payments/Late Payments, leave blank for NA): ")

    # Convert 'NA' to None or appropriate default value
    for key in customer_data:
        if customer_data[key].strip().upper() == 'NA' or customer_data[key].strip() == '':
            customer_data[key] = None

    return customer_data

# Get user input
user_data = get_user_input()

# Make a prediction
prediction = predict_fraud(user_data)

# Output the result
if prediction == 0:
    print("The loan application is predicted to be legitimate.")
else:
    print("The loan application is predicted to be non-legitimate.")
