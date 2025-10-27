"""
Data loading, cleaning, and preprocessing utilities
"""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np

from fase2.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    COLUMN_MAPPING,
    CATEGORICAL_RANGES,
    CONTINUOUS_FEATURES,
    TARGET_COL,
)

app = typer.Typer()


# ============= FUNCIONES REUTILIZABLES =============


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw German Credit dataset"""
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Data types:\n{df.dtypes.value_counts()}")
    return df


def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Translate German column names to English"""
    logger.info("Translating column names from German to English...")
    df = df.rename(columns=COLUMN_MAPPING)
    logger.success(f"✓ Translated {len(COLUMN_MAPPING)} column names")
    return df


def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Remove whitespace and problematic columns"""
    logger.info("Cleaning whitespace from all columns...")

    # Remove problematic column if exists
    if "mixed_type_col" in df.columns:
        df = df.drop(columns=["mixed_type_col"])
        logger.info("✓ Dropped 'mixed_type_col'")

    # Strip whitespace from object columns
    for col in tqdm(df.columns, desc="Cleaning columns"):
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    logger.success("✓ Whitespace removed from all columns")
    return df


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-target columns to numeric"""
    logger.info("Converting columns to numeric types...")

    numeric_cols = [col for col in df.columns if col != TARGET_COL]

    for col in tqdm(numeric_cols, desc="Converting to numeric"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.success(f"✓ Converted {len(numeric_cols)} columns to numeric")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with appropriate imputation strategy"""
    logger.info("Handling missing values...")

    missing_before = df.isnull().sum().sum()
    rows_before = len(df)

    if missing_before == 0:
        logger.success("✓ No missing values detected")
        return df

    logger.info(
        f"Found {missing_before} missing values across {(df.isnull().sum() > 0).sum()} columns"
    )

    # 1. CRÍTICO: Remove rows with missing target PRIMERO
    nans_in_target = df[TARGET_COL].isnull().sum()
    if nans_in_target > 0:
        df = df.dropna(subset=[TARGET_COL])
        logger.info(
            f"✓ Removed {nans_in_target} rows with missing target '{TARGET_COL}'"
        )

    # 2. Define variable types
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    discrete_vars = [col for col in feature_cols if col not in CONTINUOUS_FEATURES]

    # 3. Impute continuous with median
    total_imputed = 0
    for col in CONTINUOUS_FEATURES:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            n_imputed = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            total_imputed += n_imputed
            logger.info(
                f"✓ [{col}] Imputed {n_imputed} values with median ({median_val:.2f})"
            )

    # 4. Impute discrete with mode
    for col in discrete_vars:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
            n_imputed = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            total_imputed += n_imputed
            logger.info(f"✓ [{col}] Imputed {n_imputed} values with mode ({mode_val})")

    # 5. SEGURIDAD: Remove any remaining NaNs (no deberían existir pero por si acaso)
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"⚠ {remaining_missing} missing values remain across columns:")
        for col in df.columns[df.isnull().any()]:
            logger.warning(f"  - {col}: {df[col].isnull().sum()} NaNs")

        rows_before_drop = len(df)
        df = df.dropna()
        logger.warning(
            f"⚠ Dropped {rows_before_drop - len(df)} rows with remaining NaNs"
        )

    rows_after = len(df)
    logger.success(f"✓ Missing values handled:")
    logger.success(
        f"  - Rows removed: {rows_before - rows_after} ({100*(rows_before-rows_after)/rows_before:.2f}%)"
    )
    logger.success(f"  - Values imputed: {total_imputed}")
    logger.success(
        f"  - Final dataset: {rows_after} rows ({100*rows_after/rows_before:.1f}% of original)"
    )

    # VERIFICACIÓN FINAL
    final_nans = df.isnull().sum().sum()
    if final_nans > 0:
        logger.error(f"❌ ERROR: Still {final_nans} NaNs remaining!")
        raise ValueError(f"Failed to remove all NaN values. {final_nans} NaNs remain.")

    return df


def validate_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean target variable (must be 0 or 1)"""
    logger.info(f"Validating target variable '{TARGET_COL}'...")

    rows_before = len(df)

    # Mostrar valores únicos antes de limpiar
    unique_values = df[TARGET_COL].value_counts(dropna=False)
    logger.info(f"Unique values in '{TARGET_COL}' before cleaning:")
    for val, count in unique_values.items():
        logger.info(f"  - {val}: {count}")

    # Convertir a numeric (esto convierte "NAN", "invalid", "?" a NaN)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Filtrar solo valores válidos (0.0 o 1.0)
    valid_mask = df[TARGET_COL].isin([0.0, 1.0])
    invalid_count = (~valid_mask).sum()

    if invalid_count > 0:
        invalid_values = df[~valid_mask][TARGET_COL].value_counts(dropna=False)
        logger.info(f"Found {invalid_count} invalid target values:")
        for val, count in invalid_values.items():
            logger.info(f"  - {val}: {count}")

        df = df[valid_mask]
        logger.success(
            f"✓ Removed {rows_before - len(df)} rows with invalid target values"
        )
    else:
        logger.success(f"✓ All target values are valid (0 or 1)")

    # Verificar que no queden NaNs
    remaining_nans = df[TARGET_COL].isnull().sum()
    if remaining_nans > 0:
        logger.error(f"❌ Still {remaining_nans} NaNs in target after validation!")
        df = df.dropna(subset=[TARGET_COL])
        logger.info(f"✓ Removed {remaining_nans} remaining NaN rows")

    logger.success(f"✓ Target validation complete. Final dataset: {len(df)} rows")

    return df


def validate_categorical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean categorical variables"""
    logger.info("Validating categorical variables...")

    total_invalid = 0
    rows_before = len(df)

    for col, valid_values in tqdm(
        CATEGORICAL_RANGES.items(), desc="Validating categories"
    ):
        if col in df.columns:
            invalid_mask = ~df[col].isin(valid_values)
            n_invalid = invalid_mask.sum()

            if n_invalid > 0:
                logger.info(
                    f"✓ [{col}] Found {n_invalid} invalid values: {df[col][invalid_mask].unique()}"
                )
                df = df[~invalid_mask]
                total_invalid += n_invalid

    if total_invalid > 0:
        logger.success(
            f"✓ Removed {rows_before - len(df)} rows with invalid categorical values"
        )
    else:
        logger.success("✓ All categorical variables valid")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows"""
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_removed = rows_before - len(df)

    if rows_removed > 0:
        logger.info(f"✓ Removed {rows_removed} duplicate rows")
    else:
        logger.success("✓ No duplicates found")

    return df


# ============= CLI COMMAND =============


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "german_credit_modified.csv",
    output_path: Path = INTERIM_DATA_DIR / "german_credit_cleaned.csv",
):
    """
    Complete data cleaning pipeline for German Credit dataset.

    Reads raw data, applies all cleaning steps, and saves to interim directory.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    # Load
    df = load_raw_data(input_path)

    # Clean
    df = translate_columns(df)
    df = clean_whitespace(df)
    df = convert_to_numeric(df)

    # 🆕 NUEVO: Validar target ANTES de imputar
    df = validate_target_variable(df)

    df = handle_missing_values(df)
    df = validate_categorical_ranges(df)
    df = remove_duplicates(df)

    # VERIFICACIÓN FINAL antes de guardar
    logger.info("Performing final verification before saving...")
    final_nans = df.isnull().sum().sum()
    if final_nans > 0:
        logger.error(f"❌ ERROR: Dataset still has {final_nans} NaN values!")
        logger.error("NaN counts by column:")
        for col in df.columns[df.isnull().any()]:
            logger.error(f"  - {col}: {df[col].isnull().sum()} NaNs")

        # FORZAR eliminación
        logger.warning("⚠ Force-dropping all rows with any NaN...")
        rows_before = len(df)
        df = df.dropna()
        logger.warning(f"⚠ Dropped {rows_before - len(df)} rows")

    # Verificar tipos de datos
    logger.info(f"Data types before saving:\n{df.dtypes}")

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # VERIFICAR que se guardó bien
    logger.info("Verifying saved file...")
    df_test = pd.read_csv(output_path)
    test_nans = df_test.isnull().sum().sum()
    if test_nans > 0:
        logger.error(f"❌ File saved WITH NaNs! ({test_nans} NaNs found)")
    else:
        logger.success(f"✓ File verified clean (0 NaNs)")

    logger.success("=" * 60)
    logger.success(f"✓ CLEANED DATA SAVED TO: {output_path}")
    logger.success(f"✓ Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.success("=" * 60)


if __name__ == "__main__":
    app()
