
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv(
    r"C:\Users\yasas\OneDrive\Desktop\customer churn\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Dataset loaded:", df.shape)

# 2. Remove customer ID
df.drop(columns=["customerID"], inplace=True, errors="ignore")

# 3. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 4. Prepare target and features
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Churn"])

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

# 5. Identify column types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# 6. Preprocess numeric columns
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# 7. Preprocess categorical columns
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 8. Build model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 9. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 10. Train model
print("Training model...")
model.fit(X_train, y_train)

# 11. Evaluate model
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 12. Save complete pipeline
with open("churn_pipeline.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")