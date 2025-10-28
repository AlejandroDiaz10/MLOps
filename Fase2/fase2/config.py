"""
Configuration management using Pydantic for validation
"""

from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class PathConfig(BaseModel):
    """Path configurations for the project"""

    proj_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Optional[Path] = None
    raw_data_dir: Optional[Path] = None
    interim_data_dir: Optional[Path] = None
    processed_data_dir: Optional[Path] = None
    external_data_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    reports_dir: Optional[Path] = None
    figures_dir: Optional[Path] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-configure derived paths
        self.data_dir = self.proj_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.interim_data_dir = self.data_dir / "interim"
        self.processed_data_dir = self.data_dir / "processed"
        self.external_data_dir = self.data_dir / "external"
        self.models_dir = self.proj_root / "models"
        self.reports_dir = self.proj_root / "reports"
        self.figures_dir = self.reports_dir / "figures"


class ModelConfig(BaseModel):
    """Model training and evaluation configurations"""

    random_state: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Test set proportion"
    )
    cv_folds: int = Field(default=5, ge=2, le=10, description="Cross-validation folds")
    auc_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Minimum AUC-ROC threshold"
    )

    @validator("test_size")
    def validate_test_size(cls, v):
        """Ensure test_size is in valid range"""
        if not 0.0 < v < 1.0:
            raise ValueError("test_size must be between 0 and 1 (exclusive)")
        return v

    @validator("cv_folds")
    def validate_cv_folds(cls, v):
        """Ensure cv_folds is reasonable"""
        if v < 2:
            raise ValueError("cv_folds must be at least 2")
        if v > 20:
            logger.warning(f"cv_folds={v} is very high, this may slow down training")
        return v


class DataConfig(BaseModel):
    """Data processing configurations"""

    target_col: str = Field(
        default="credit_risk", description="Name of target variable"
    )

    continuous_features: List[str] = Field(
        default=[
            "duration",
            "amount",
            "installment_rate",
            "age",
            "residence_duration",
            "existing_credits",
        ],
        description="List of continuous features for special handling",
    )

    categorical_ranges: Dict[str, List[float]] = Field(
        default={
            "checking_account": [1.0, 2.0, 3.0, 4.0],
            "credit_history": [0.0, 1.0, 2.0, 3.0, 4.0],
            "purpose": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0],
            "savings_account": [1.0, 2.0, 3.0, 4.0, 5.0],
            "employment_duration": [1.0, 2.0, 3.0, 4.0, 5.0],
            "personal_status": [1.0, 2.0, 3.0, 4.0],
            "other_debtors": [1.0, 2.0, 3.0],
            "property": [1.0, 2.0, 3.0, 4.0],
            "other_installment_plans": [1.0, 2.0, 3.0],
            "housing": [1.0, 2.0, 3.0],
            "job": [1.0, 2.0, 3.0, 4.0],
            "dependents": [1.0, 2.0],
            "telephone": [1.0, 2.0],
            "foreign_worker": [1.0, 2.0],
        },
        description="Valid value ranges for categorical variables",
    )

    column_mapping: Dict[str, str] = Field(
        default={
            "laufkont": "checking_account",
            "laufzeit": "duration",
            "moral": "credit_history",
            "verw": "purpose",
            "hoehe": "amount",
            "sparkont": "savings_account",
            "beszeit": "employment_duration",
            "rate": "installment_rate",
            "famges": "personal_status",
            "buerge": "other_debtors",
            "wohnzeit": "residence_duration",
            "verm": "property",
            "alter": "age",
            "weitkred": "other_installment_plans",
            "wohn": "housing",
            "bishkred": "existing_credits",
            "beruf": "job",
            "pers": "dependents",
            "telef": "telephone",
            "gastarb": "foreign_worker",
            "kredit": "credit_risk",
        },
        description="Mapping from German to English column names",
    )


class Config(BaseModel):
    """Main configuration class that aggregates all configs"""

    paths: PathConfig = Field(default_factory=PathConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        logger.info(f"Configuration initialized. Project root: {self.paths.proj_root}")

    def validate_paths(self) -> bool:
        """Validate that critical paths exist"""
        critical_paths = [
            self.paths.data_dir,
            self.paths.raw_data_dir,
        ]

        for path in critical_paths:
            if not path.exists():
                logger.warning(f"Path does not exist: {path}")
                return False

        return True


# Singleton instance - import this in other modules
config = Config()

# Legacy exports for backward compatibility
PROJ_ROOT = config.paths.proj_root
DATA_DIR = config.paths.data_dir
RAW_DATA_DIR = config.paths.raw_data_dir
INTERIM_DATA_DIR = config.paths.interim_data_dir
PROCESSED_DATA_DIR = config.paths.processed_data_dir
EXTERNAL_DATA_DIR = config.paths.external_data_dir
MODELS_DIR = config.paths.models_dir
REPORTS_DIR = config.paths.reports_dir
FIGURES_DIR = config.paths.figures_dir

RANDOM_STATE = config.model.random_state
TEST_SIZE = config.model.test_size
CV_FOLDS = config.model.cv_folds
AUC_THRESHOLD = config.model.auc_threshold

TARGET_COL = config.data.target_col
CONTINUOUS_FEATURES = config.data.continuous_features
CATEGORICAL_RANGES = config.data.categorical_ranges
COLUMN_MAPPING = config.data.column_mapping
