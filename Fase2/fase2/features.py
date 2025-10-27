"""
Feature engineering and preprocessing utilities
"""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from fase2.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    CONTINUOUS_FEATURES,
)

app = typer.Typer()


# ============= FUNCIONES REUTILIZABLES =============


def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> tuple:
    """
    Detect outliers using IQR method
    Returns cleaned DataFrame and outlier mask
    """
    logger.info(f"Detecting outliers in {len(columns)} columns using IQR method...")

    outlier_mask = pd.Series([False] * len(df), index=df.index)
    outlier_counts = {}

    for col in tqdm(columns, desc="Detecting outliers"):
        if col not in df.columns:
            logger.warning(f"⚠ Column '{col}' not found in dataframe, skipping...")
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_counts[col] = col_outliers.sum()
        outlier_mask = outlier_mask | col_outliers

    total_outliers = outlier_mask.sum()
    df_clean = df[~outlier_mask].copy()

    # IMPORTANTE: Verificar NaNs después de filtrar
    nans_after = df_clean.isnull().sum().sum()
    if nans_after > 0:
        logger.warning(
            f"⚠ Found {nans_after} NaN values after outlier removal, dropping them..."
        )
        df_clean = df_clean.dropna()

    logger.info(f"✓ Found {total_outliers} rows with outliers")
    for col, count in outlier_counts.items():
        if count > 0:
            logger.info(f"  - {col}: {count} outliers")

    logger.success(
        f"✓ Removed {total_outliers} outlier rows ({100*total_outliers/len(df):.2f}%)"
    )
    logger.success(f"✓ Final clean shape: {df_clean.shape}")

    return df_clean, outlier_mask


def verify_no_nans(df: pd.DataFrame, step_name: str = "dataset"):
    """Verify that dataframe has no NaN values"""
    nans = df.isnull().sum()
    total_nans = nans.sum()

    if total_nans > 0:
        logger.error(f"❌ NaN values found in {step_name}:")
        for col in nans[nans > 0].index:
            logger.error(f"  - {col}: {nans[col]} NaNs")
        raise ValueError(f"Input contains NaN values. Found {total_nans} NaNs total.")
    else:
        logger.success(f"✓ No NaN values in {step_name}")


def split_features_target(df: pd.DataFrame) -> tuple:
    """Separate features and target variable"""
    logger.info(f"Separating features and target variable '{TARGET_COL}'...")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logger.success(f"✓ Features shape: {X.shape}, Target shape: {y.shape}")

    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Split data into train and test sets"""
    logger.info(
        f"Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.success(f"✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    logger.info(f"  - Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"  - Test class distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Standardize features using StandardScaler
    Fit on training, transform both
    """
    logger.info("Scaling features using StandardScaler...")

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    logger.success("✓ Features scaled successfully")

    return X_train_scaled, X_test_scaled, scaler


# ============= CLI COMMAND =============


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "german_credit_cleaned.csv",
    train_features_path: Path = PROCESSED_DATA_DIR / "X_train.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "X_test.csv",
    train_labels_path: Path = PROCESSED_DATA_DIR / "y_train.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "y_test.csv",
    scaler_path: Path = MODELS_DIR / "scaler.pkl",
):
    """
    Complete feature engineering pipeline.

    Reads cleaned data, handles outliers, splits data, scales features,
    and saves processed datasets ready for modeling.
    """
    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # Load cleaned data
    logger.info(f"Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded shape: {df.shape}")

    # VERIFICACIÓN 1: Check NaNs after loading
    verify_no_nans(df, "loaded data")

    # Detect and remove outliers
    outlier_columns = [
        col for col in df.columns if col != TARGET_COL and col in CONTINUOUS_FEATURES
    ]
    logger.info(f"Columns to check for outliers: {outlier_columns}")
    df_clean, _ = detect_outliers_iqr(df, outlier_columns)

    # VERIFICACIÓN 2: Check NaNs after outlier removal
    verify_no_nans(df_clean, "after outlier removal")

    # Split features and target
    X, y = split_features_target(df_clean)

    # VERIFICACIÓN 3: Check NaNs in features
    verify_no_nans(X, "features X")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # VERIFICACIÓN 4: Check NaNs in splits
    verify_no_nans(X_train, "X_train")
    verify_no_nans(X_test, "X_test")

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save everything
    logger.info("Saving processed datasets...")

    train_features_path.parent.mkdir(parents=True, exist_ok=True)
    test_features_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_csv(train_features_path, index=False)
    X_test_scaled.to_csv(test_features_path, index=False)
    y_train.to_csv(train_labels_path, index=False, header=["credit_risk"])
    y_test.to_csv(test_labels_path, index=False, header=["credit_risk"])
    joblib.dump(scaler, scaler_path)

    logger.success("=" * 60)
    logger.success("✓ FEATURE ENGINEERING COMPLETE")
    logger.success(f"✓ Train features saved to: {train_features_path}")
    logger.success(f"✓ Test features saved to: {test_features_path}")
    logger.success(f"✓ Train labels saved to: {train_labels_path}")
    logger.success(f"✓ Test labels saved to: {test_labels_path}")
    logger.success(f"✓ Scaler saved to: {scaler_path}")
    logger.success(f"✓ Final dataset sizes:")
    logger.success(f"   - X_train: {X_train_scaled.shape}")
    logger.success(f"   - X_test: {X_test_scaled.shape}")
    logger.success("=" * 60)


if __name__ == "__main__":
    app()
