from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    # Model settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    AUC_THRESHOLD = 0.75

    # Target column
    TARGET_COL = "credit_risk"

    # Categorical variable ranges
    CATEGORICAL_RANGES = {
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
    }

    # Feature groups
    CONTINUOUS_FEATURES = [
        "duration",
        "amount",
        "installment_rate",
        "age",
        "residence_duration",
        "existing_credits",
    ]

    # Column mapping (German to English)
    COLUMN_MAPPING = {
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
    }

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
