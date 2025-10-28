"""
Factory pattern for creating ML models with configurations
"""

from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from loguru import logger

from fase2.config import config


class ModelFactory:
    """
    Factory for creating ML models with standard configurations.
    Implements the Factory design pattern.
    """

    # Default hyperparameter grids for each model type
    DEFAULT_PARAM_GRIDS = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "logistic_regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "decision_tree": {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    }

    @staticmethod
    def create_model(model_name: str, **kwargs) -> Any:
        """
        Create a model instance by name

        Args:
            model_name: Name of model ('random_forest', 'logistic_regression', 'decision_tree')
            **kwargs: Additional parameters to pass to model constructor

        Returns:
            Instantiated model object

        Raises:
            ValueError: If model_name is unknown
        """
        random_state = kwargs.pop("random_state", config.model.random_state)

        if model_name == "random_forest":
            logger.debug(
                f"Creating Random Forest model with random_state={random_state}"
            )
            return RandomForestClassifier(random_state=random_state, **kwargs)

        elif model_name == "logistic_regression":
            logger.debug(
                f"Creating Logistic Regression model with random_state={random_state}"
            )
            return LogisticRegression(
                random_state=random_state, max_iter=1000, **kwargs
            )

        elif model_name == "decision_tree":
            logger.debug(
                f"Creating Decision Tree model with random_state={random_state}"
            )
            return DecisionTreeClassifier(random_state=random_state, **kwargs)

        else:
            available = list(ModelFactory.DEFAULT_PARAM_GRIDS.keys())
            raise ValueError(
                f"Unknown model type: '{model_name}'. Available: {available}"
            )

    @staticmethod
    def get_param_grid(model_name: str, custom_grid: Optional[Dict] = None) -> Dict:
        """
        Get hyperparameter grid for model

        Args:
            model_name: Name of model
            custom_grid: Optional custom parameter grid to use instead of default

        Returns:
            Parameter grid dictionary
        """
        if custom_grid is not None:
            logger.info(f"Using custom parameter grid for {model_name}")
            return custom_grid

        if model_name not in ModelFactory.DEFAULT_PARAM_GRIDS:
            raise ValueError(f"No default param grid for model: {model_name}")

        logger.debug(f"Using default parameter grid for {model_name}")
        return ModelFactory.DEFAULT_PARAM_GRIDS[model_name]

    @staticmethod
    def get_available_models() -> list:
        """Get list of available model names"""
        return list(ModelFactory.DEFAULT_PARAM_GRIDS.keys())
