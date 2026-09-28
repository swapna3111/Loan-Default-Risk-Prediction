import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv("data/loan_clean.csv")

print("Dataset loaded!")
print("Shape:", df.shape)

# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("\nTarget distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. Identify numerical and categorical columns
# --------------------------------------------------

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumerical columns:", list(numerical_features))
print("\nCategorical columns:", list(categorical_features))

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 6. Create Logistic Regression model
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 9. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL RESULTS ==========")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

joblib.dump(model, "loan_default_model.pkl")

print("\nModel saved as loan_default_model.pkl")