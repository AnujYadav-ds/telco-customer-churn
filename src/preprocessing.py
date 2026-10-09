import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# Function 1 : load_data
def load_data(file_path):
    """
    Load the Telco Customer Churn dataset.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df
# Function2 : Clean_data
def clean_data(df):
    """
    Clean the Telco Customer Churn dataset.

    Parameters:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.drop("customerID", axis=1)

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    return df
# Function 3 : Engineered_features
def engineer_features(df):
    """
    Create engineered features for the Telco Customer Churn dataset.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """
    df = df.copy()

    # Tenure group
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=[
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    # Number of services
    df["num_services"] = (
        df["PhoneService"].eq("Yes").astype(int)
        + df["MultipleLines"].eq("Yes").astype(int)
        + df["InternetService"].ne("No").astype(int)
        + df["OnlineSecurity"].eq("Yes").astype(int)
        + df["OnlineBackup"].eq("Yes").astype(int)
        + df["DeviceProtection"].eq("Yes").astype(int)
        + df["TechSupport"].eq("Yes").astype(int)
        + df["StreamingTV"].eq("Yes").astype(int)
        + df["StreamingMovies"].eq("Yes").astype(int)
    )

    return df
# Function 4 : Preparing feature target
def prepare_features_target(df):
    """
    Separate features and target variable.

    Parameters:
        df (pd.DataFrame): Dataset containing the target column.

    Returns:
        tuple:
            X (pd.DataFrame): Feature variables.
            y (pd.Series): Target variable.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y
# Function 5 : Split data
def split_data(X, y):
    """
    Split features and target into stratified training and testing sets.

    Parameters:
        X (pd.DataFrame): Feature variables.
        y (pd.Series): Target variable.

    Returns:
        tuple:
            X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
# Function 6 : Identifying feature types
def identify_feature_types(X):
    """
    Identify numeric and categorical feature columns.

    Parameters:
        X (pd.DataFrame): Feature dataset.

    Returns:
        tuple:
            numeric_features (list): Numeric feature column names.
            categorical_features (list): Categorical feature column names.
    """
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numeric_features, categorical_features
# Function 7 : Building preprocessing pipeline
def build_preprocessor(numeric_features, categorical_features):
    """
    Build a preprocessing pipeline for numeric and categorical features.

    Parameters:
        numeric_features (list): Numeric feature column names.
        categorical_features (list): Categorical feature column names.

    Returns:
        ColumnTransformer: Preprocessing transformer.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor