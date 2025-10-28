"""
Pipeline builder for creating scikit-learn pipelines.
Implements best practices for ML workflow automation.
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from loguru import logger

from fase2.config import config
from fase2.transformers import OutlierRemover, TypeConverter, CategoricalValidator
from fase2.core.model_factory import ModelFactory


class PipelineBuilder:
    """
    Builder for creating scikit-learn pipelines with preprocessing and modeling.

    This class follows the Builder pattern to construct complete ML pipelines
    that include all preprocessing steps and the final estimator.

    Example:
        >>> builder = PipelineBuilder()
        >>> pipeline = builder.build_pipeline(model_name='random_forest')
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """

    def __init__(self, config_obj=None):
        """
        Initialize PipelineBuilder.

        Args:
            config_obj: Optional configuration object
        """
        self.config = config_obj or config
        logger.debug("PipelineBuilder initialized")

    def build_preprocessing_pipeline(self) -> Pipeline:
        """
        Build preprocessing pipeline without the model.

        This pipeline includes:
        1. Type conversion
        2. Categorical validation
        3. Outlier removal
        4. Imputation (median for continuous, mode for discrete)
        5. Scaling (StandardScaler)

        Returns:
            Preprocessing pipeline
        """
        logger.info("Building preprocessing pipeline...")

        # Identify feature types
        continuous_features = [
            f
            for f in self.config.data.continuous_features
            if f != self.config.data.target_col
        ]

        # Create separate transformers for continuous and discrete features
        continuous_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        discrete_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),  # Scale discrete features too
            ]
        )

        # Note: This assumes all columns will be passed
        # In practice, you'd identify discrete columns dynamically

        preprocessing_pipeline = Pipeline(
            steps=[
                (
                    "type_converter",
                    TypeConverter(exclude_columns=[self.config.data.target_col]),
                ),
                ("categorical_validator", CategoricalValidator()),
                ("outlier_remover", OutlierRemover(columns=continuous_features)),
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),  # Simplified for all features
                ("scaler", StandardScaler()),
            ]
        )

        logger.success("✓ Preprocessing pipeline built")
        return preprocessing_pipeline

    def build_pipeline(
        self,
        model_name: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        include_outlier_clipping: bool = False,
    ) -> Pipeline:
        """
        Build complete pipeline with preprocessing and model.

        IMPORTANT: This pipeline assumes data is already cleaned.
        It only handles:
        - Outlier clipping (optional)
        - Imputation
        - Scaling
        - Model training

        Args:
            model_name: Name of model to use
            model_params: Optional model hyperparameters
            include_outlier_clipping: Whether to include outlier clipping

        Returns:
            Complete scikit-learn Pipeline
        """
        logger.info(f"Building sklearn pipeline for {model_name}...")

        # Get model
        model = ModelFactory.create_model(model_name, **(model_params or {}))

        # Build preprocessing steps (SIMPLIFIED)
        steps = []

        # Optional: Outlier clipping (usually not needed if data already cleaned)
        if include_outlier_clipping:
            continuous_features = [
                f
                for f in self.config.data.continuous_features
                if f != self.config.data.target_col
            ]
            steps.append(
                (
                    "outlier_clipper",
                    OutlierRemover(columns=continuous_features, remove_rows=False),
                )
            )

        # Core preprocessing (safe transformations)
        steps.extend(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

        pipeline = Pipeline(steps=steps)

        logger.success(f"✓ Pipeline built with {len(steps)} steps")
        for step_name, transformer in pipeline.steps:
            logger.debug(f"  - {step_name}: {type(transformer).__name__}")

        return pipeline

    def build_grid_search_pipeline(
        self,
        model_name: str = "random_forest",
        param_grid: Optional[Dict] = None,
        cv_folds: Optional[int] = None,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> GridSearchCV:
        """
        Build pipeline with GridSearchCV for hyperparameter tuning.

        Args:
            model_name: Name of model to use
            param_grid: Parameter grid for GridSearch (uses default if None)
            cv_folds: Number of CV folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            verbose: Verbosity level

        Returns:
            GridSearchCV object with pipeline
        """
        logger.info(f"Building GridSearchCV pipeline for {model_name}...")

        cv_folds = cv_folds or self.config.model.cv_folds

        # Build base pipeline (simplified, no data cleaning)
        base_pipeline = self.build_pipeline(
            model_name, include_outlier_clipping=False  # Don't clip during CV
        )

        # Get parameter grid
        if param_grid is None:
            default_grid = ModelFactory.get_param_grid(model_name)
            # Prefix params with 'model__' for pipeline
            param_grid = {f"model__{k}": v for k, v in default_grid.items()}

        logger.info(f"Parameter grid: {list(param_grid.keys())}")
        logger.info(f"Cross-validation: {cv_folds} folds")

        # Create GridSearchCV with error handling
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            error_score="raise",  # 🆕 Raise errors for debugging
        )

        logger.success("✓ GridSearchCV pipeline built")
        return grid_search

    def get_pipeline_steps(self, pipeline: Pipeline) -> pd.DataFrame:
        """
        Get a summary of pipeline steps.

        Args:
            pipeline: Scikit-learn pipeline

        Returns:
            DataFrame with step information
        """
        steps_info = []

        for name, transformer in pipeline.steps:
            steps_info.append(
                {
                    "Step": name,
                    "Transformer": type(transformer).__name__,
                    "Parameters": str(transformer.get_params())[:100] + "...",
                }
            )

        return pd.DataFrame(steps_info)


def create_full_pipeline(
    model_name: str = "random_forest", use_grid_search: bool = True
) -> Pipeline:
    """
    Convenience function to create a complete pipeline.

    Args:
        model_name: Model to use
        use_grid_search: Whether to use GridSearchCV

    Returns:
        Complete pipeline (with or without GridSearch)
    """
    builder = PipelineBuilder()

    if use_grid_search:
        return builder.build_grid_search_pipeline(model_name)
    else:
        return builder.build_pipeline(model_name)
