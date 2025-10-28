"""
Pipeline builder for creating scikit-learn pipelines.
Implements best practices for ML workflow automation.
"""

from typing import Optional, Dict, Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from loguru import logger

from fase2.config import config
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

    def build_pipeline(
        self,
        model_name: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """
        Build complete pipeline with preprocessing and model.

        IMPORTANT: This pipeline assumes data is already cleaned.
        It only handles:
        - Imputation (backup for any remaining NaN)
        - Scaling
        - Model training

        Args:
            model_name: Name of model to use
            model_params: Optional model hyperparameters

        Returns:
            Complete scikit-learn Pipeline
        """
        logger.info(f"Building sklearn pipeline for {model_name}...")

        # Get model
        model = ModelFactory.create_model(model_name, **(model_params or {}))

        # Build preprocessing steps (SIMPLIFIED - data already cleaned)
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]

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
        base_pipeline = self.build_pipeline(model_name)

        # Get parameter grid
        if param_grid is None:
            default_grid = ModelFactory.get_param_grid(model_name)
            # Prefix params with 'model__' for pipeline
            param_grid = {f"model__{k}": v for k, v in default_grid.items()}

        logger.info(f"Parameter grid: {list(param_grid.keys())}")
        logger.info(f"Cross-validation: {cv_folds} folds")

        # Create GridSearchCV
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            error_score="raise",
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
