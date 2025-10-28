"""
Demonstration of scikit-learn Pipeline best practices.

This script shows how to use the complete sklearn pipeline
for reproducible and automated ML workflows.
"""

import pandas as pd
from loguru import logger

from fase2.pipeline_builder import PipelineBuilder, create_full_pipeline
from fase2.config import config


def demo_pipeline():
    """Demonstrate sklearn pipeline usage."""

    logger.info("=" * 70)
    logger.info("SKLEARN PIPELINE DEMONSTRATION")
    logger.info("=" * 70)

    # Load PROCESSED data (not raw!)
    logger.info("\n1️⃣ Loading processed data...")

    # Check if processed data exists
    X_train_path = config.paths.processed_data_dir / "X_train.csv"
    y_train_path = config.paths.processed_data_dir / "y_train.csv"

    if not X_train_path.exists() or not y_train_path.exists():
        logger.error("Processed data not found!")
        logger.info("Please run data preparation first:")
        logger.info("  python -m fase2.dataset")
        logger.info("  python -m fase2.features")
        return None

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    X_test = pd.read_csv(config.paths.processed_data_dir / "X_test.csv")
    y_test = pd.read_csv(config.paths.processed_data_dir / "y_test.csv").values.ravel()

    logger.success(f"✓ Data loaded: Train={X_train.shape}, Test={X_test.shape}")

    # Build pipeline
    logger.info("\n2️⃣ Building sklearn Pipeline...")
    builder = PipelineBuilder()

    # Simple pipeline (no GridSearch for demo)
    simple_pipeline = builder.build_pipeline("random_forest")

    # Display pipeline structure
    logger.info("\n📋 Pipeline Structure:")
    steps_df = builder.get_pipeline_steps(simple_pipeline)
    print(steps_df.to_string(index=False))

    # Train pipeline
    logger.info("\n3️⃣ Training pipeline...")
    simple_pipeline.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_pred = simple_pipeline.predict(X_test)
    y_proba = simple_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.success("\n✓ Pipeline trained!")
    logger.success(f"  Test Accuracy: {accuracy:.4f}")
    logger.success(f"  Test AUC-ROC: {auc:.4f}")

    # Make predictions
    logger.info("\n4️⃣ Sample predictions:")
    for i in range(5):
        logger.info(
            f"  Sample {i+1}: Class={int(y_pred[i])}, Probability={y_proba[i]:.4f}, True={int(y_test[i])}"
        )

    logger.info("\n" + "=" * 70)
    logger.success("✓ DEMONSTRATION COMPLETE")
    logger.info("=" * 70)

    return simple_pipeline


if __name__ == "__main__":
    demo_pipeline()
