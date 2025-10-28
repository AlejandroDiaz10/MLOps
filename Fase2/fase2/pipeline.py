"""
Main ML Pipeline orchestrator using OOP design.
Implements the Template Method pattern.
"""

from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from fase2.config import config
from fase2.core.data_processor import DataProcessor
from fase2.core.feature_engineer import FeatureEngineer
from fase2.core.trainer import ModelTrainer
from fase2.core.evaluator import ModelEvaluator
from fase2.exceptions import PipelineError


class MLPipeline:
    """
    Main ML Pipeline that orchestrates all steps.

    This class provides a high-level API to run the complete ML workflow
    from data loading to model evaluation.

    Example:
        >>> pipeline = MLPipeline()
        >>> pipeline.run_full_pipeline(model_name='random_forest')

    Attributes:
        config: Configuration object
        data_processor: DataProcessor instance
        feature_engineer: FeatureEngineer instance
        trainer: ModelTrainer instance
        evaluator: ModelEvaluator instance
    """

    def __init__(self, config_obj=None):
        """
        Initialize pipeline with configuration

        Args:
            config_obj: Optional custom configuration object
        """
        self.config = config_obj or config
        self.data_processor = DataProcessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)

        logger.info("MLPipeline initialized")

    def run_data_preparation(
        self, input_path: Optional[Path] = None, output_path: Optional[Path] = None
    ) -> Path:
        """
        Run complete data preparation pipeline.

        Steps:
        1. Load raw data
        2. Translate column names
        3. Clean whitespace
        4. Convert to numeric
        5. Validate target
        6. Handle missing values
        7. Validate categorical ranges
        8. Remove duplicates
        9. Save cleaned data

        Args:
            input_path: Path to raw data CSV
            output_path: Path to save cleaned data

        Returns:
            Path to cleaned data file
        """
        logger.info("=" * 70)
        logger.info("STEP 1: DATA PREPARATION PIPELINE")
        logger.info("=" * 70)

        if output_path is None:
            output_path = (
                self.config.paths.interim_data_dir / "german_credit_cleaned.csv"
            )

        try:
            # Method chaining for clean, declarative API
            self.data_processor.load_raw_data(
                input_path
            ).translate_columns().clean_whitespace().convert_to_numeric().validate_target().handle_missing_values().validate_categorical_ranges().remove_duplicates().save(
                output_path
            )

            # Print summary
            summary = self.data_processor.get_summary()
            logger.info("\n📊 Data Preparation Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")

            logger.success("=" * 70)
            logger.success("✓ DATA PREPARATION COMPLETE")
            logger.success(f"✓ Output: {output_path}")
            logger.success("=" * 70)

            return output_path

        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise PipelineError(f"Data preparation failed: {str(e)}")

    def run_feature_engineering(
        self, input_path: Optional[Path] = None, output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Run complete feature engineering pipeline.

        Steps:
        1. Load cleaned data
        2. Detect and remove outliers
        3. Split features and target
        4. Train-test split
        5. Scale features
        6. Save all processed data

        Args:
            input_path: Path to cleaned data
            output_dir: Directory to save processed data

        Returns:
            Dictionary with paths to all saved files
        """
        logger.info("=" * 70)
        logger.info("STEP 2: FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 70)

        if input_path is None:
            input_path = (
                self.config.paths.interim_data_dir / "german_credit_cleaned.csv"
            )

        try:
            paths = (
                self.feature_engineer.load_data(input_path)
                .detect_outliers()
                .split_target()
                .train_test_split()
                .scale_features()
                .save_all(output_dir)
            )

            # Print summary
            summary = self.feature_engineer.get_summary()
            logger.info("\n📊 Feature Engineering Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")

            logger.success("=" * 70)
            logger.success("✓ FEATURE ENGINEERING COMPLETE")
            logger.success("✓ Outputs:")
            for name, path in paths.items():
                logger.success(f"  {name}: {path}")
            logger.success("=" * 70)

            return paths

        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise PipelineError(f"Feature engineering failed: {str(e)}")

    def run_training(
        self,
        model_name: str = "random_forest",
        param_grid: Optional[Dict] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Run model training pipeline.

        Steps:
        1. Load training data
        2. Train model with GridSearchCV
        3. Evaluate with cross-validation
        4. Save model and metadata

        Args:
            model_name: Name of model to train
            param_grid: Optional custom hyperparameter grid
            output_dir: Directory to save model

        Returns:
            Path to saved model file
        """
        logger.info("=" * 70)
        logger.info(f"STEP 3: MODEL TRAINING - {model_name.upper()}")
        logger.info("=" * 70)

        try:
            model_path = (
                self.trainer.load_training_data()
                .train(model_name, param_grid=param_grid)
                .evaluate()
                .save(output_dir)
            )

            logger.success("=" * 70)
            logger.success("✓ MODEL TRAINING COMPLETE")
            logger.success(f"✓ Model saved: {model_path}")
            logger.success("=" * 70)

            return model_path

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise PipelineError(f"Model training failed: {str(e)}")

    def run_evaluation(
        self, model_path: Optional[Path] = None, save_results: bool = True
    ) -> Dict:
        """
        Run model evaluation pipeline.

        Steps:
        1. Load trained model
        2. Load test data
        3. Make predictions
        4. Evaluate performance
        5. Save predictions and metrics (optional)
        6. Print classification report

        Args:
            model_path: Path to trained model
            save_results: Whether to save predictions and metrics

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 70)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 70)

        try:
            self.evaluator.load_model(model_path).load_test_data().predict().evaluate()

            if save_results:
                self.evaluator.save_predictions()
                self.evaluator.save_metrics()

            self.evaluator.print_classification_report()

            metrics = self.evaluator.get_metrics()

            logger.success("=" * 70)
            logger.success("✓ MODEL EVALUATION COMPLETE")
            logger.success("=" * 70)

            return metrics

        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise PipelineError(f"Model evaluation failed: {str(e)}")

    def run_full_pipeline(
        self,
        model_name: str = "random_forest",
        param_grid: Optional[Dict] = None,
        skip_data_prep: bool = False,
        skip_feature_eng: bool = False,
        generate_plots: bool = True,
    ) -> Dict:
        """
        Run the complete ML pipeline from start to finish.

        This is the main entry point for running the entire workflow.

        Args:
            model_name: Name of model to train
            param_grid: Optional custom hyperparameter grid
            skip_data_prep: Skip data preparation if already done
            skip_feature_eng: Skip feature engineering if already done
            generate_plots: Generate visualization plots

        Returns:
            Dictionary with all pipeline results
        """
        logger.info("\n")
        logger.info("🚀" * 35)
        logger.info("🚀  STARTING COMPLETE ML PIPELINE")
        logger.info("🚀" * 35)
        logger.info("\n")

        results = {}

        try:
            # Step 1: Data Preparation
            if not skip_data_prep:
                cleaned_path = self.run_data_preparation()
                results["cleaned_data_path"] = cleaned_path
            else:
                logger.info(
                    "⏭️  Skipping data preparation (using existing cleaned data)"
                )

            # Step 2: Feature Engineering
            if not skip_feature_eng:
                feature_paths = self.run_feature_engineering()
                results["feature_paths"] = feature_paths
            else:
                logger.info(
                    "⏭️  Skipping feature engineering (using existing processed data)"
                )

            # Step 3: Model Training
            model_path = self.run_training(model_name, param_grid)
            results["model_path"] = model_path

            # Step 4: Model Evaluation
            metrics = self.run_evaluation(model_path)
            results["metrics"] = metrics

            # Step 5: Generate Visualizations
            if generate_plots:
                figures = self.run_visualization(model_path)
                results["figures"] = figures

            # Final summary
            logger.info("\n")
            logger.info("🎉" * 35)
            logger.info("🎉  PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("🎉" * 35)
            logger.info("\n")
            logger.info("📊 Final Results Summary:")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Test AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
            logger.info(f"  Model saved to: {model_path}")
            if generate_plots:
                logger.info(f"  Figures generated: {len(results.get('figures', {}))}")
            logger.info("\n")

            return results

        except Exception as e:
            logger.error("\n")
            logger.error("❌" * 35)
            logger.error(f"❌  PIPELINE FAILED: {str(e)}")
            logger.error("❌" * 35)
            raise PipelineError(f"Pipeline execution failed: {str(e)}")

    def run_multiple_models(
        self,
        model_names: list = None,
        skip_data_prep: bool = False,
        skip_feature_eng: bool = False,
        generate_plots: bool = True,
    ) -> Dict[str, Dict]:
        """
        Train and evaluate multiple models for comparison.

        Args:
            model_names: List of model names to train
            skip_data_prep: Skip data preparation
            skip_feature_eng: Skip feature engineering
            generate_plots: Generate comparison visualizations

        Returns:
            Dictionary mapping model names to their results
        """
        if model_names is None:
            from fase2.core.model_factory import ModelFactory

            model_names = ModelFactory.get_available_models()

        logger.info(f"\n🔄 Training {len(model_names)} models: {model_names}\n")

        # Data prep and feature engineering (once)
        if not skip_data_prep:
            self.run_data_preparation()

        if not skip_feature_eng:
            self.run_feature_engineering()

        # Load test data for predictions (needed for plots)
        import pandas as pd

        X_test_path = self.config.paths.processed_data_dir / "X_test.csv"
        y_test_path = self.config.paths.processed_data_dir / "y_test.csv"
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()

        # Train each model
        all_results = {}

        for model_name in model_names:
            logger.info(f"\n{'='*70}")
            logger.info(
                f"Training model {len(all_results)+1}/{len(model_names)}: {model_name}"
            )
            logger.info(f"{'='*70}\n")

            try:
                # Train
                model_path = self.run_training(model_name)

                # Evaluate
                metrics = self.run_evaluation(model_path, save_results=False)

                # Load model for additional info
                import joblib

                model = joblib.load(model_path)

                # Get predictions and probabilities
                y_pred = model.predict(X_test)
                y_proba = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Get CV scores from trainer if available
                cv_scores = None
                if (
                    hasattr(self.trainer, "cv_scores")
                    and self.trainer.cv_scores is not None
                ):
                    cv_scores = self.trainer.cv_scores.tolist()

                # Get feature importance
                feature_importance = None
                if hasattr(model, "feature_importances_"):
                    feature_importance = pd.DataFrame(
                        {
                            "Feature": X_test.columns,
                            "Importance": model.feature_importances_,
                        }
                    ).sort_values("Importance", ascending=False)

                all_results[model_name] = {
                    "model_path": model_path,
                    "metrics": metrics,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "cv_scores": cv_scores,
                    "feature_importance": feature_importance,
                }

                # Generate individual plots
                if generate_plots:
                    self.run_visualization(model_path, save_individual=True)

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                all_results[model_name] = {"error": str(e)}

        # Print comparison
        self._print_model_comparison(all_results)

        # Generate comparison plots
        if generate_plots:
            logger.info("\n")
            self.run_comparison_visualizations(all_results)

        return all_results

    def _print_model_comparison(self, results: Dict[str, Dict]):
        """Print comparison table of multiple models"""
        logger.info("\n")
        logger.info("📊" * 35)
        logger.info("📊  MODEL COMPARISON")
        logger.info("📊" * 35)
        logger.info("\n")

        # Create comparison table
        print(
            f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10}"
        )
        print("-" * 85)

        for model_name, result in results.items():
            if "error" in result:
                print(f"{model_name:<25} ERROR: {result['error']}")
            else:
                m = result["metrics"]
                auc = f"{m.get('auc_roc', 0):.4f}" if m.get("auc_roc") else "N/A"
                print(
                    f"{model_name:<25} {m['accuracy']:.4f}     {m['precision']:.4f}     "
                    f"{m['recall']:.4f}     {m['f1_score']:.4f}     {auc}"
                )

        print("\n")

    def run_visualization(
        self,
        model_path: Optional[Path] = None,
        X_test_path: Optional[Path] = None,
        y_test_path: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Generate visualizations for model evaluation.

        Args:
            model_path: Path to trained model
            X_test_path: Path to test features
            y_test_path: Path to test labels

        Returns:
            Dictionary with paths to generated figures
        """
        logger.info("=" * 70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 70)

        from fase2.plots import (
            plot_confusion_matrix,
            plot_roc_curve,
            plot_feature_importance,
        )
        import pandas as pd
        import joblib

        # Load data
        if model_path is None:
            model_path = self.config.paths.models_dir / "random_forest.pkl"
        if X_test_path is None:
            X_test_path = self.config.paths.processed_data_dir / "X_test.csv"
        if y_test_path is None:
            y_test_path = self.config.paths.processed_data_dir / "y_test.csv"

        model = joblib.load(model_path)
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Generate plots
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)

        model_name = model_path.stem.replace("_", " ").title()
        figures = {}

        # Confusion Matrix
        figures["confusion_matrix"] = plot_confusion_matrix(cm, model_name)

        # ROC Curve
        if y_proba is not None:
            figures["roc_curve"] = plot_roc_curve(y_test, y_proba, model_name)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {"Feature": X_test.columns, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False)
            figures["feature_importance"] = plot_feature_importance(
                importance_df, model_name
            )

        logger.success("✓ Visualizations generated:")
        for name, path in figures.items():
            logger.info(f"  {name}: {path}")

        return figures

    def run_visualization(
        self, model_path: Optional[Path] = None, save_individual: bool = True
    ) -> Dict[str, Path]:
        """
        Generate visualizations for a single model.

        Args:
            model_path: Path to trained model
            save_individual: Whether to save individual model plots

        Returns:
            Dictionary with paths to generated figures
        """
        import joblib
        import pandas as pd
        from sklearn.metrics import confusion_matrix
        from fase2.plots import (
            plot_confusion_matrix,
            plot_roc_curve,
            plot_feature_importance,
        )

        logger.info("=" * 70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 70)

        # Default paths
        if model_path is None:
            model_path = self.config.paths.models_dir / "random_forest.pkl"

        X_test_path = self.config.paths.processed_data_dir / "X_test.csv"
        y_test_path = self.config.paths.processed_data_dir / "y_test.csv"

        # Load data
        model = joblib.load(model_path)
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        model_name = model_path.stem.replace("_", " ").title()
        figures = {}

        if save_individual:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            figures["confusion_matrix"] = plot_confusion_matrix(cm, model_name)

            # ROC Curve
            if y_proba is not None:
                figures["roc_curve"] = plot_roc_curve(y_test, y_proba, model_name)

            # Feature Importance
            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame(
                    {
                        "Feature": X_test.columns,
                        "Importance": model.feature_importances_,
                    }
                ).sort_values("Importance", ascending=False)
                figures["feature_importance"] = plot_feature_importance(
                    importance_df, model_name
                )

        logger.success(f"✓ Generated {len(figures)} plots for {model_name}")

        return figures

    def run_comparison_visualizations(
        self, results: Dict[str, Dict], save_path: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Generate comparison visualizations for multiple models.

        Args:
            results: Dictionary with model results from run_multiple_models
            save_path: Directory to save plots

        Returns:
            Dictionary with paths to comparison plots
        """
        from fase2.plots import (
            plot_model_comparison,
            plot_multiple_roc_curves,
            plot_training_history,
            plot_target_distribution,
        )
        import pandas as pd

        logger.info("=" * 70)
        logger.info("GENERATING COMPARISON VISUALIZATIONS")
        logger.info("=" * 70)

        if save_path is None:
            save_path = self.config.paths.figures_dir

        save_path.mkdir(parents=True, exist_ok=True)

        # Load test data
        y_test_path = self.config.paths.processed_data_dir / "y_test.csv"
        y_test = pd.read_csv(y_test_path).values.ravel()

        # Load cleaned data for target distribution
        cleaned_path = self.config.paths.interim_data_dir / "german_credit_cleaned.csv"
        df_clean = pd.read_csv(cleaned_path) if cleaned_path.exists() else None

        figures = {}

        # Target distribution
        if df_clean is not None:
            try:
                figures["target_distribution"] = plot_target_distribution(
                    df_clean, save_path=save_path
                )
            except Exception as e:
                logger.warning(f"Could not plot target distribution: {e}")

        # Model comparison
        try:
            figures["model_comparison"] = plot_model_comparison(
                results, save_path=save_path
            )
        except Exception as e:
            logger.error(f"Failed to plot model comparison: {e}")

        # Multiple ROC curves
        try:
            figures["roc_comparison"] = plot_multiple_roc_curves(
                results, y_test, save_path=save_path
            )
        except Exception as e:
            logger.error(f"Failed to plot ROC comparison: {e}")

        # CV scores comparison
        try:
            cv_plot = plot_training_history(results, save_path=save_path)
            if cv_plot:
                figures["cv_comparison"] = cv_plot
        except Exception as e:
            logger.error(f"Failed to plot CV comparison: {e}")

        logger.success(f"✓ Generated {len(figures)} comparison plots")

        return figures
