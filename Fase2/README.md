# Fase 2: German Credit Risk Prediction

[![Python 3.12.1](https://img.shields.io/badge/python-3.12.1-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyecto académico de Machine Learning para predecir riesgo crediticio utilizando el dataset German Credit, estructurado siguiendo mejores prácticas de MLOps con Cookiecutter Data Science y programación orientada a objetos.

**Equipo:** Team 34  
**Curso:** Machine Learning  
**Fecha:** Octubre 2025

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Opción 1: Pipeline Completo](#opción-1-pipeline-completo-recomendado)
  - [Opción 2: Paso por Paso](#opción-2-paso-por-paso)
  - [Opción 3: Comparar Modelos](#opción-3-comparar-múltiples-modelos)
  - [Opción 4: Uso Programático](#opción-4-uso-programático)
- [Visualizaciones](#visualizaciones)
- [Notebooks Interactivos](#notebooks-interactivos)
- [Tecnologías](#tecnologías)
- [Aprendizajes Clave](#aprendizajes-clave)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema de predicción de riesgo crediticio utilizando el dataset **German Credit**. El objetivo es clasificar clientes como "buen crédito" (1) o "mal crédito" (0) basándose en características demográficas y financieras.

**Características principales:**
- ✅ Arquitectura orientada a objetos (OOP)
- ✅ Pipeline de ML completo y modular
- ✅ Estructura estandarizada con Cookiecutter Data Science
- ✅ Configuración validada con Pydantic
- ✅ Method chaining para API limpia
- ✅ Exception handling robusto
- ✅ Visualizaciones automáticas
- ✅ Código reutilizable y mantenible
- ✅ Backward compatibility con código funcional

**Métrica objetivo:** AUC-ROC ≥ 0.75

---

## 🏗️ Arquitectura

El proyecto implementa una **arquitectura en capas** con separación de responsabilidades:
```
┌─────────────────────────────────────────────┐
│  CAPA 1: CLI / Scripts                      │  ← Interfaz de usuario
│  (run_pipeline.py, dataset.py, etc.)        │
├─────────────────────────────────────────────┤
│  CAPA 2: Pipeline Orchestrator              │  ← Coordinación
│  (MLPipeline)                               │
├─────────────────────────────────────────────┤
│  CAPA 3: Core Business Logic (OOP)          │  ← Lógica principal
│  (DataProcessor, FeatureEngineer,           │
│   ModelTrainer, ModelEvaluator)             │
├─────────────────────────────────────────────┤
│  CAPA 4: Outputs & Artifacts                │  ← Resultados
│  (models/, reports/, data/processed/)       │
└─────────────────────────────────────────────┘
```

### **Clases Principales**

| Clase | Responsabilidad | Patrón |
|-------|----------------|--------|
| `DataProcessor` | Limpieza y validación de datos | Method Chaining |
| `FeatureEngineer` | Ingeniería de características | Method Chaining |
| `ModelTrainer` | Entrenamiento con GridSearchCV | Template Method |
| `ModelEvaluator` | Evaluación y predicción | - |
| `ModelFactory` | Creación de modelos | Factory Pattern |
| `MLPipeline` | Orquestación del workflow | Facade Pattern |

### **Backward Compatibility**

El proyecto mantiene **dos APIs simultáneas** para máxima flexibilidad:

**API Funcional (Legacy):**
```python
from fase2.config import RAW_DATA_DIR, TARGET_COL
from fase2.dataset import load_raw_data
```

**API Orientada a Objetos (Recomendada):**
```python
from fase2.config import config
from fase2.core.data_processor import DataProcessor
```

Ambas APIs coexisten sin conflictos, permitiendo:
- ✅ Notebooks antiguos siguen funcionando
- ✅ Nuevo código usa OOP
- ✅ Migración gradual sin romper funcionalidad

---

## 📁 Estructura del Proyecto
```
Fase2/
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_pipeline.py          # 🆕 Script principal
│
├── data/
│   ├── raw/                 # Datos originales (inmutables)
│   ├── interim/             # Datos limpiados
│   ├── processed/           # Datos finales para modelado
│   └── external/
│
├── models/                  # Modelos (.pkl) y metadata (.json)
│
├── notebooks/               # Análisis exploratorio
│   ├── 1.0-t34-data-exploration.ipynb
│   ├── 2.0-t34-feature-engineering.ipynb
│   └── 3.0-t34-model-training.ipynb
│
├── reports/
│   └── figures/             # Visualizaciones generadas
│
└── fase2/                   # 🆕 Código fuente (OOP)
    ├── config.py            # Configuración con Pydantic
    ├── exceptions.py        # Custom exceptions
    ├── pipeline.py          # MLPipeline orchestrator
    ├── plots.py             # Funciones de visualización
    │
    ├── core/                # Lógica de negocio
    │   ├── data_processor.py
    │   ├── feature_engineer.py
    │   ├── model_factory.py
    │   ├── trainer.py
    │   └── evaluator.py
    │
    ├── utils/               # Utilidades
    │   ├── logger.py
    │   └── validators.py
    │
    ├── dataset.py           # CLI para limpieza (wrapper)
    ├── features.py          # CLI para features (wrapper)
    └── modeling/
        ├── train.py         # CLI para entrenamiento (wrapper)
        └── predict.py       # CLI para predicción (wrapper)
```

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.12.1+
- pip
- virtualenv (recomendado)

### Pasos
```bash
# 1. Clonar repositorio
git clone https://github.com/AlejandroDiaz10/MLOps
cd Fase2

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar paquete en modo desarrollo
pip install -e .

# 5. Verificar instalación
python -c "from fase2 import config; print('✓ Installation successful')"
```

---

## 💻 Uso

### **Opción 1: Pipeline Completo** (Recomendado) 🚀

Ejecuta el workflow completo con un comando:
```bash
# Pipeline completo: limpieza → features → entrenamiento → evaluación → plots
python run_pipeline.py full
```

**Con opciones:**
```bash
# Modelo específico
python run_pipeline.py full --model-name logistic_regression

# Saltar pasos ya completados
python run_pipeline.py full --skip-data-prep --skip-feature-eng
```

**Output generado:**
- `data/interim/german_credit_cleaned.csv`
- `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `models/random_forest.pkl` + `random_forest_metadata.json`
- `data/processed/test_predictions.csv`, `test_metrics.json`
- `reports/figures/confusion_matrix_random_forest.png`
- `reports/figures/roc_curve_random_forest.png`
- `reports/figures/feature_importance_random_forest.png`

---

### **Opción 2: Paso por Paso**

Ejecuta cada etapa individualmente para debugging:
```bash
# 1. Limpieza de datos
python -m fase2.dataset
# Output: data/interim/german_credit_cleaned.csv

# 2. Feature Engineering
python -m fase2.features
# Output: data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv

# 3. Entrenamiento
python -m fase2.modeling.train --model-name random_forest
# Output: models/random_forest.pkl + metadata

# 4. Evaluación
python -m fase2.modeling.predict --model-path models/random_forest.pkl
# Output: test_predictions.csv, test_metrics.json
```

---

### **Opción 3: Comparar Múltiples Modelos**

Entrena y compara todos los modelos automáticamente:
```bash
# Entrenar y comparar: random_forest, logistic_regression, decision_tree
python run_pipeline.py compare
```

**Visualizaciones generadas:**

**Por cada modelo:**
- `confusion_matrix_<model>.png`
- `roc_curve_<model>.png`
- `feature_importance_<model>.png`

**Comparaciones generales:**
- `target_distribution.png` - Distribución de clases
- `model_comparison.png` - Métricas lado a lado
- `roc_curves_comparison.png` - Todas las ROC juntas
- `cv_scores_comparison.png` - Cross-validation scores

**Comandos útiles:**
```bash
# Ver modelos disponibles
python run_pipeline.py models

# Comparar con datos ya procesados
python run_pipeline.py compare --skip-data-prep --skip-feature-eng
```

---

### **Opción 4: Uso Programático**

#### **A) API de Alto Nivel (MLPipeline)**
```python
from fase2.pipeline import MLPipeline

# Pipeline completo
pipeline = MLPipeline()
results = pipeline.run_full_pipeline(model_name='random_forest')

# Acceder a resultados
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"AUC-ROC: {results['metrics']['auc_roc']:.4f}")
print(f"Model: {results['model_path']}")

# Ejecutar paso por paso
pipeline.run_data_preparation()
pipeline.run_feature_engineering()
pipeline.run_training('logistic_regression')
metrics = pipeline.run_evaluation()
figures = pipeline.run_visualization()

# Comparar modelos
results = pipeline.run_multiple_models(
    model_names=['random_forest', 'logistic_regression'],
    generate_plots=True
)
```

#### **B) API de Bajo Nivel (Clases Individuales)**
```python
from fase2.core.data_processor import DataProcessor
from fase2.core.feature_engineer import FeatureEngineer
from fase2.core.trainer import ModelTrainer
from fase2.core.evaluator import ModelEvaluator

# 1. Procesamiento de datos con method chaining
processor = DataProcessor()
df_clean = processor \
    .load_raw_data() \
    .translate_columns() \
    .clean_whitespace() \
    .convert_to_numeric() \
    .validate_target() \
    .handle_missing_values() \
    .validate_categorical_ranges() \
    .remove_duplicates() \
    .get_data()

# 2. Feature engineering
engineer = FeatureEngineer()
paths = engineer \
    .load_data() \
    .detect_outliers() \
    .split_target() \
    .train_test_split() \
    .scale_features() \
    .save_all()

X_train, X_test, y_train, y_test = engineer.get_train_test_split()

# 3. Entrenamiento
trainer = ModelTrainer()
model_path = trainer \
    .load_training_data() \
    .train('random_forest') \
    .evaluate() \
    .save()

# 4. Evaluación
evaluator = ModelEvaluator()
metrics = evaluator \
    .load_model(model_path) \
    .load_test_data() \
    .predict() \
    .evaluate() \
    .save_predictions() \
    .save_metrics() \
    .get_metrics()

evaluator.print_classification_report()
```

#### **C) API Funcional (Legacy - Notebooks)**
```python
# Importaciones legacy (siguen funcionando)
from fase2.config import RAW_DATA_DIR, TARGET_COL
from fase2.dataset import load_raw_data, translate_columns
from fase2.features import scale_features

# Usar funciones directamente (como antes)
df = load_raw_data()
df = translate_columns(df)
# ... etc
```

---

## 📊 Visualizaciones

El pipeline genera automáticamente visualizaciones profesionales en `reports/figures/`:

### **Visualizaciones por Modelo**

| Plot | Descripción | Archivo |
|------|-------------|---------|
| Confusion Matrix | Matriz de confusión con porcentajes | `confusion_matrix_<model>.png` |
| ROC Curve | Curva ROC con AUC score | `roc_curve_<model>.png` |
| Feature Importance | Top 15 features más importantes | `feature_importance_<model>.png` |

### **Visualizaciones Comparativas**

| Plot | Descripción | Archivo |
|------|-------------|---------|
| Target Distribution | Distribución de clases (balanceo) | `target_distribution.png` |
| Model Comparison | Métricas lado a lado (bar chart) | `model_comparison.png` |
| ROC Comparison | Todas las curvas ROC juntas | `roc_curves_comparison.png` |
| CV Scores | Cross-validation performance | `cv_scores_comparison.png` |

**Características:**
- ✅ Resolución 300 DPI (publicación)
- ✅ Colores profesionales (viridis, seaborn)
- ✅ Anotaciones con valores numéricos
- ✅ Threshold lines para AUC ≥ 0.75
- ✅ Formato PNG optimizado

---

## 📓 Notebooks Interactivos

Los notebooks en `notebooks/` demuestran análisis exploratorio:

1. **`1.0-t34-data-exploration.ipynb`**
   - Análisis exploratorio del dataset
   - Estadísticas descriptivas
   - Distribuciones de features

2. **`2.0-t34-feature-engineering.ipynb`**
   - Exploración de ingeniería de features
   - Detección de outliers
   - Análisis de correlaciones

3. **`3.0-t34-model-training.ipynb`**
   - Entrenamiento de modelos
   - Comparación de performance
   - Análisis de resultados

### **Setup para Notebooks**
```python
# Habilitar autoreload (recarga automática de módulos)
%load_ext autoreload
%autoreload 2

# Importar API OOP (recomendado)
from fase2.pipeline import MLPipeline
from fase2.core.data_processor import DataProcessor

# Usar
pipeline = MLPipeline()
processor = DataProcessor()
df = processor.load_raw_data().translate_columns().get_data()

# API funcional también disponible
from fase2.config import INTERIM_DATA_DIR
import pandas as pd
df = pd.read_csv(INTERIM_DATA_DIR / 'german_credit_cleaned.csv')
```

---

## 🛠️ Tecnologías

**Core:**
- Python 3.12.1
- pandas 2.1.0 - Manipulación de datos
- numpy 1.24.3 - Cálculos numéricos
- scikit-learn 1.3.0 - Modelos ML

**Validación y Configuración:**
- pydantic - Validación de esquemas
- loguru - Logging estructurado
- typer - CLI moderna

**Visualización:**
- matplotlib 3.7.2
- seaborn 0.12.2

**Desarrollo:**
- Cookiecutter Data Science - Estructura
- Git - Control de versiones
- Jupyter - Análisis interactivo
- joblib - Serialización

---

## 🎓 Aprendizajes Clave

Este proyecto demuestra implementación profesional de:

### **1. Programación Orientada a Objetos**
- ✅ Clases con responsabilidades únicas (SRP)
- ✅ Encapsulación de estado y comportamiento
- ✅ Method chaining para API fluida
- ✅ Type hints completos

### **2. Design Patterns**
- ✅ **Factory Pattern** - `ModelFactory` para creación de modelos
- ✅ **Template Method** - `MLPipeline` para workflow estándar
- ✅ **Facade Pattern** - Interfaz simplificada para sistema complejo

### **3. Software Engineering Best Practices**
- ✅ **Separation of Concerns** - Arquitectura en capas
- ✅ **DRY Principle** - Código reutilizable
- ✅ **Configuration as Code** - Pydantic para validación
- ✅ **Error Handling** - Custom exceptions descriptivas
- ✅ **Backward Compatibility** - API dual (funcional + OOP)

### **4. MLOps Practices**
- ✅ Pipeline reproducible
- ✅ Versionamiento de modelos (metadata JSON)
- ✅ Logging estructurado
- ✅ Configuración centralizada
- ✅ Visualizaciones automáticas

---

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

---

## 📚 Referencias

- **Dataset:** [UCI Machine Learning Repository - German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Template:** [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- **Documentación Scikit-learn:** [scikit-learn.org](https://scikit-learn.org/)


---

**Proyecto académico con fines educativos - Octubre 2025**