import pandas as pd
import numpy as np
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
        'Loan Purpose': random.choice(loan_purposes[loan_type]),  # Correctly mapped purpose
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

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 20)       # Show first 20 rows
pd.set_option('display.width', 1000)        # Avoid wrapping issues
pd.set_option('display.colheader_justify', 'left')

# Display first 10 rows
print(df.head(10))

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoder for later use

scaler = StandardScaler()
num_cols = ['Age', 'CIBIL Score', 'Loan Amount Requested', 'Loan Tenure',
            'Existing Loans', 'DTI Ratio', 'Credit Utilization Ratio',
            'Monthly Income', 'Total Savings & Deposits',
            'Number of Credit Cards', 'Total Credit Limit']

df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop(columns=['Fraud'])
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== Model Evaluation on Test Set ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)
print("Updated Classification Report:\n", classification_report(y_test, y_pred))

final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {final_accuracy:.2f}")

y_pred = model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy After Resampling: {final_accuracy:.2f}")

print("=== Final Model Evaluation ===")
print(f"Final Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))