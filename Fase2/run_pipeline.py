"""
Main entry point for the complete ML pipeline.

This script orchestrates the entire workflow following best practices:
1. Data cleaning (dataset.py)
2. Feature engineering (features.py)
3. Model training with sklearn Pipeline (modeling/train.py)
4. Model evaluation (modeling/predict.py)

Usage:
    python run_pipeline.py                              # Run complete pipeline
    python run_pipeline.py --model-name logistic_regression
    python run_pipeline.py --skip-data-prep --skip-feature-eng
"""

import typer
from pathlib import Path
from loguru import logger

from fase2.dataset import main as run_data_cleaning
from fase2.features import main as run_feature_engineering
from fase2.modeling.train import train_model
from fase2.modeling.predict import evaluate_model
from fase2.config import config

app = typer.Typer(help="German Credit Risk ML Pipeline - Complete workflow automation")


@app.command()
def run(
    model_name: str = typer.Option(
        "random_forest",
        "--model-name",
        "-m",
        help="Model to train: random_forest, logistic_regression, decision_tree",
    ),
    skip_data_prep: bool = typer.Option(
        False,
        "--skip-data-prep",
        help="Skip data preparation step (use existing cleaned data)",
    ),
    skip_feature_eng: bool = typer.Option(
        False,
        "--skip-feature-eng",
        help="Skip feature engineering step (use existing processed data)",
    ),
    cv_folds: int = typer.Option(
        None,
        "--cv-folds",
        help="Number of cross-validation folds (default from config)",
    ),
    no_plots: bool = typer.Option(
        False, "--no-plots", help="Skip generating visualization plots"
    ),
):
    """
    Run complete ML pipeline end-to-end.

    Stages:
    1. Data Cleaning (Etapa 1)
    2. Feature Engineering (Etapa 2)
    3. Model Training with sklearn Pipeline (Etapa 3)
    4. Model Evaluation and Visualization

    Examples:
        # Complete pipeline with Random Forest
        python run_pipeline.py

        # With Logistic Regression
        python run_pipeline.py --model-name logistic_regression

        # With Decision Tree
        python run_pipeline.py --model-name decision_tree

        # Skip preprocessing (data already prepared)
        python run_pipeline.py --skip-data-prep --skip-feature-eng

        # Fast training (fewer CV folds)
        python run_pipeline.py --cv-folds 3
    """
    logger.info("\n")
    logger.info("🚀" * 35)
    logger.info("🚀  GERMAN CREDIT RISK - ML PIPELINE")
    logger.info("🚀" * 35)
    logger.info("\n")

    try:
        # Step 1: Data Preparation
        if not skip_data_prep:
            logger.info("=" * 70)
            logger.info("STEP 1: DATA PREPARATION")
            logger.info("=" * 70)
            run_data_cleaning()
        else:
            logger.info("⏭️  Skipping data preparation (using existing data)")

        # Step 2: Feature Engineering
        if not skip_feature_eng:
            logger.info("\n")
            logger.info("=" * 70)
            logger.info("STEP 2: FEATURE ENGINEERING")
            logger.info("=" * 70)
            run_feature_engineering()
        else:
            logger.info("⏭️  Skipping feature engineering (using existing data)")

        # Step 3: Model Training
        logger.info("\n")
        logger.info("=" * 70)
        logger.info(f"STEP 3: MODEL TRAINING - {model_name.upper()}")
        logger.info("=" * 70)
        model_path = train_model(model_name=model_name, cv_folds=cv_folds)

        # Step 4: Model Evaluation
        logger.info("\n")
        logger.info("=" * 70)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 70)
        metrics = evaluate_model(model_path=model_path, generate_plots=not no_plots)

        # Final Summary
        logger.info("\n")
        logger.info("🎉" * 35)
        logger.info("🎉  PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("🎉" * 35)
        logger.info("\n")
        logger.info("📊 Final Results:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Test AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
        logger.info(f"  Model saved: {model_path}")
        logger.info("\n")
        logger.info("📁 Generated files:")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Figures: {config.paths.figures_dir}")
        logger.info(f"  - Metrics: {config.paths.processed_data_dir}")

    except Exception as e:
        logger.error("\n")
        logger.error("❌" * 35)
        logger.error(f"❌  PIPELINE FAILED: {str(e)}")
        logger.error("❌" * 35)
        logger.exception("Full error traceback:")
        raise typer.Exit(code=1)


@app.command()
def compare(
    models: str = typer.Option(
        "random_forest,logistic_regression,decision_tree",
        "--models",
        help="Comma-separated list of models to compare",
    ),
    cv_folds: int = typer.Option(
        3, "--cv-folds", help="Number of CV folds (reduced for speed when comparing)"
    ),
    skip_prep: bool = typer.Option(
        False,
        "--skip-prep",
        help="Skip data preparation for all models (must already be processed)",
    ),
):
    """
    Train and compare multiple models.

    Examples:
        # Compare all 3 models
        python run_pipeline.py compare

        # Compare only 2 models
        python run_pipeline.py compare --models random_forest,logistic_regression

        # Use existing processed data
        python run_pipeline.py compare --skip-prep
    """

    model_list = [m.strip() for m in models.split(",")]

    logger.info("\n")
    logger.info("🔬" * 35)
    logger.info(f"🔬  COMPARING {len(model_list)} MODELS")
    logger.info("🔬" * 35)
    logger.info(f"Models: {', '.join(model_list)}")
    logger.info("\n")

    results = {}

    # Data preparation (only once)
    if not skip_prep:
        logger.info("=" * 70)
        logger.info("STEP 1: DATA PREPARATION (once for all models)")
        logger.info("=" * 70)
        run_data_cleaning()

        logger.info("\n")
        logger.info("=" * 70)
        logger.info("STEP 2: FEATURE ENGINEERING (once for all models)")
        logger.info("=" * 70)
        run_feature_engineering()
    else:
        logger.info("⏭️  Skipping data preparation (using existing data)")

    # Train each model
    for idx, model_name in enumerate(model_list, 1):
        logger.info("\n")
        logger.info("=" * 70)
        logger.info(f"MODEL {idx}/{len(model_list)}: {model_name.upper()}")
        logger.info("=" * 70)

        try:
            # Train
            model_path = train_model(model_name=model_name, cv_folds=cv_folds)

            # Evaluate
            metrics = evaluate_model(model_path=model_path, generate_plots=True)

            results[model_name] = {"metrics": metrics, "model_path": model_path}

            logger.success(
                f"✓ {model_name}: "
                f"AUC={metrics.get('auc_roc', 0):.4f}, "
                f"Acc={metrics['accuracy']:.4f}"
            )

        except Exception as e:
            logger.error(f"✗ {model_name} failed: {str(e)}")
            results[model_name] = {"error": str(e)}

    # Summary comparison
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("📊 MODEL COMPARISON SUMMARY")
    logger.info("=" * 70)

    comparison_data = []
    for model_name, result in results.items():
        if "metrics" in result:
            metrics = result["metrics"]
            comparison_data.append(
                {
                    "Model": model_name.replace("_", " ").title(),
                    "AUC-ROC": metrics.get("auc_roc", 0),
                    "Accuracy": metrics.get("accuracy", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1-Score": metrics.get("f1_score", 0),
                }
            )

    # Print comparison table
    if comparison_data:
        # Sort by AUC-ROC
        comparison_data.sort(key=lambda x: x["AUC-ROC"], reverse=True)

        logger.info("\n")
        logger.info(
            f"{'Model':<25} {'AUC-ROC':<10} {'Accuracy':<10} "
            f"{'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
        )
        logger.info("-" * 85)

        for row in comparison_data:
            logger.info(
                f"{row['Model']:<25} "
                f"{row['AUC-ROC']:<10.4f} "
                f"{row['Accuracy']:<10.4f} "
                f"{row['Precision']:<10.4f} "
                f"{row['Recall']:<10.4f} "
                f"{row['F1-Score']:<10.4f}"
            )

        logger.info("\n")
        best_model = comparison_data[0]["Model"]
        best_auc = comparison_data[0]["AUC-ROC"]
        logger.success(f"🏆 Best model: {best_model} (AUC-ROC: {best_auc:.4f})")

        logger.info("\n")
        logger.info("📁 All results saved:")
        logger.info(f"  - Models: {config.paths.models_dir}")
        logger.info(f"  - Figures: {config.paths.figures_dir}")
    else:
        logger.warning("No models completed successfully")


@app.command()
def clean():
    """
    Clean generated files (models, processed data, figures).

    Example:
        python run_pipeline.py clean
    """

    logger.warning("\n🗑️  Cleaning generated files...")

    import shutil

    dirs_to_clean = [
        (config.paths.models_dir, "*.pkl"),
        (config.paths.models_dir, "*.json"),
        (config.paths.processed_data_dir, "*.csv"),
        (config.paths.figures_dir, "*.png"),
        (config.paths.interim_data_dir, "*.csv"),
    ]

    file_count = 0
    for dir_path, pattern in dirs_to_clean:
        if dir_path.exists():
            for item in dir_path.glob(pattern):
                if item.is_file():
                    item.unlink()
                    file_count += 1
                    logger.info(f"   Deleted: {item.name}")

    if file_count > 0:
        logger.success(f"\n✅ Cleaned {file_count} files!")
    else:
        logger.info("\n✓ No files to clean")


if __name__ == "__main__":
    app()
