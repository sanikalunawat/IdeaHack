import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

np.random.seed(42)
size = 10000  # Number of transactions

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

# Save dataset
df.to_csv("credit_card_fraud_data.csv", index=False)

# Check fraud class distribution
print(df["Fraudulent"].value_counts())

# Plot class distribution
sns.countplot(x=df["Fraudulent"])
plt.title("Fraud vs. Non-Fraud Distribution")
plt.show()

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target (y)
X = df.drop(columns=["Fraudulent"])
y = df["Fraudulent"]

# Split dataset into training and testing sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution before SMOTE
print("Before SMOTE:", y_train.value_counts())

# Apply SMOTE for oversampling
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("After SMOTE:", y_train_resampled.value_counts())

smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

# Print Classification Report
print("=== Model Evaluation ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 20)       # Show first 20 rows
pd.set_option('display.width', 1000)        # Avoid wrapping issues
pd.set_option('display.colheader_justify', 'left')

# Display first 10 rows
print(df.head(10))