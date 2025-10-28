"""
Custom exceptions for the Fase2 project
"""


class Fase2Exception(Exception):
    """Base exception for all custom exceptions in this project"""

    pass


class DataLoadError(Fase2Exception):
    """Raised when data cannot be loaded from file"""

    pass


class DataValidationError(Fase2Exception):
    """Raised when data validation fails"""

    pass


class FeatureEngineeringError(Fase2Exception):
    """Raised when feature engineering process fails"""

    pass


class ModelTrainingError(Fase2Exception):
    """Raised when model training fails"""

    pass


class ModelNotFoundError(Fase2Exception):
    """Raised when a model file cannot be found"""

    pass


class ConfigurationError(Fase2Exception):
    """Raised when configuration is invalid"""

    pass


class PipelineError(Fase2Exception):
    """Raised when pipeline execution fails"""

    pass
