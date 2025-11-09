# German Credit Risk Prediction

[![Python 3.12.1](https://img.shields.io/badge/python-3.12.1-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema de clasificación de riesgo crediticio implementando mejores prácticas de MLOps: estructura Cookiecutter, programación orientada a objetos, sklearn Pipeline y experiment tracking.

**Equipo:** Team 34 | **Curso:** Machine Learning | **Fecha:** Octubre 2025

---

## 🎯 Quickstart

```bash
# 1. Clonar e instalar
git clone https://github.com/AlejandroDiaz10/MLOps
cd Fase3
pip install -r requirements.txt && pip install -e .

# 2. Entrenar modelo
python run_pipeline.py

# 3. Comparar modelos
python run_pipeline.py compare
```

---

## 📊 Resultados

**Mejores modelos (Test Set):**

| Modelo | AUC-ROC | Accuracy | F1-Score |
|--------|---------|----------|----------|
| Random Forest | **0.8196** | 0.7464 | 0.8416 |
| Logistic Regression | 0.8186 | 0.7971 | 0.8716 |
| Decision Tree | 0.7680 | 0.7681 | 0.8431 |

**Meta alcanzada:** AUC-ROC ≥ 0.75 ✅

---

## 🏗️ Implementación de Mejores Prácticas

Este proyecto implementa las **4 etapas** de desarrollo profesional en Machine Learning:

### **Etapa 1: Estructuración del Proyecto**

**Objetivo:** Organización estandarizada del código usando Cookiecutter Data Science.

**Implementación:**
```
Fase3/
├── run_pipeline.py             # 🎯 Script principal
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── raw/                    # Datos originales
│   ├── interim/                # Datos limpiados
│   └── processed/              # Train/test splits
│
├── models/                     # Modelos .pkl + metadata
│
├── reports/figures/            # Confusion matrix, ROC curves
│
├── notebooks/                  # Análisis exploratorio
│   ├── 1.0-t34-data-exploration.ipynb
│   ├── 2.0-t34-feature-engineering.ipynb
│   ├── 3.0-t34-model-training.ipynb
│   └── 4.0-t34-sklearn-pipeline-best-practices.ipynb
│
└── fase3/                      # Código fuente
    ├── config.py               # Configuración (Pydantic)
    ├── pipeline_builder.py     # sklearn Pipeline builder
    ├── plots.py                # Visualizaciones
    │
    ├── core/                   # Lógica de negocio (OOP)
    │   ├── data_processor.py
    │   ├── feature_engineer.py
    │   └── model_factory.py
    │
    └── modeling/
        ├── train.py            # Entrenamiento
        └── predict.py          # Evaluación
```

**Beneficios:**
- ✅ Separación clara entre datos raw y procesados
- ✅ Código reproducible y versionable
- ✅ Facilita colaboración en equipo

---

### **Etapa 2: Refactorización con OOP**

**Objetivo:** Código mantenible, escalable y testeable mediante programación orientada a objetos.

**Componentes principales:**

| Clase | Responsabilidad | Patrón de Diseño |
|-------|----------------|------------------|
| `DataProcessor` | Limpieza y validación de datos | Method Chaining |
| `FeatureEngineer` | Feature engineering y splits | Method Chaining |
| `ModelFactory` | Creación de modelos ML | Factory Pattern |
| `PipelineBuilder` | Construcción de pipelines | Builder Pattern |
| `ModelTrainer` | Entrenamiento y validación | Template Method |
| `ModelEvaluator` | Evaluación de modelos | - |

**Ejemplo de uso:**
```python
from fase3.core.data_processor import DataProcessor

# Method chaining para limpieza
processor = DataProcessor()
df_clean = (processor
    .load_raw_data()
    .translate_columns()
    .clean_whitespace()
    .convert_to_numeric()
    .validate_target()
    .handle_missing_values()
    .remove_duplicates()
    .get_data())
```

**Principios SOLID aplicados:**
- **Single Responsibility:** Cada clase tiene una única responsabilidad
- **Open/Closed:** Extensible sin modificar código existente
- **Dependency Inversion:** Configuración inyectable

**Beneficios:**
- ✅ Código modular y reutilizable
- ✅ Fácil mantenimiento a largo plazo
- ✅ Testing simplificado

---

### **Etapa 3: sklearn Pipeline & Best Practices**

**Objetivo:** Automatizar preprocesamiento, entrenamiento y evaluación de forma reproducible.

**Arquitectura del Pipeline:**
```python
sklearn.Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])
```

**Ventajas sobre código tradicional:**

| Sin Pipeline | Con sklearn Pipeline |
|-------------|---------------------|
| ❌ Data leakage (fit en todo el dataset) | ✅ Fit solo en train set |
| ❌ Múltiples objetos a serializar | ✅ Un solo .pkl |
| ❌ Preprocessing manual en producción | ✅ Preprocessing automático |
| ❌ Difícil reproducir experimentos | ✅ Reproducible por diseño |

**Implementación:**
```python
from fase3.pipeline_builder import PipelineBuilder

# Construcción automática del pipeline
builder = PipelineBuilder()
grid_pipeline = builder.build_grid_search_pipeline(
    model_name='random_forest',
    cv_folds=5
)

# Entrenamiento con GridSearchCV
grid_pipeline.fit(X_train, y_train)

# Serialización (un solo objeto)
joblib.dump(grid_pipeline.best_estimator_, 'model.pkl')

# Producción (carga y predice)
model = joblib.load('model.pkl')
predictions = model.predict(X_new)  # Preprocessing incluido
```

**Best practices implementadas:**
- ✅ Cross-validation con GridSearchCV
- ✅ Búsqueda de hiperparámetros automática
- ✅ No data leakage (fit/transform separados)
- ✅ Reproducibilidad (`random_state=42`)
- ✅ Pipeline serializable completo

**Beneficios:**
- ✅ Código listo para producción
- ✅ Reducción de errores humanos
- ✅ Experimentos reproducibles

---

### **Etapa 4: Experiment Tracking & Model Management** ⏳

**Objetivo:** Tracking sistemático de experimentos, versionamiento de modelos y comparación de resultados.

**Herramientas a implementar:**

**1. MLflow**
- [ ] Tracking de parámetros e hiperparámetros
- [ ] Logging de métricas (AUC, accuracy, F1)
- [ ] Registro de artifacts (modelos, plots)
- [ ] Model Registry con versionamiento
- [ ] Comparación visual de experimentos
- [ ] Transición de stages (Dev → Staging → Production)

**2. DVC (Data Version Control)**
- [ ] Versionamiento de datasets
- [ ] Versionamiento de modelos
- [ ] Pipelines reproducibles
- [ ] Remote storage (Google Drive/S3)

**Estructura prevista:**
```bash
# MLflow tracking
mlflow ui  # Dashboard en localhost:5000

# DVC versioning
dvc init
dvc remote add -d storage gdrive://...
dvc add data/raw/german_credit_modified.csv
dvc push
```

**Beneficios esperados:**
- ✅ Comparación sistemática de experimentos
- ✅ Rollback a versiones anteriores
- ✅ Auditoría completa de modelos
- ✅ Colaboración eficiente en equipo

---

## ⚡ Ejecución de Pipelines

### **Pipeline Completo (Recomendado)**

```bash
# Entrenar con Random Forest (default)
python run_pipeline.py
```

**Salida:**
```
🚀 GERMAN CREDIT RISK - ML PIPELINE
======================================================================
STEP 1: DATA PREPARATION       # Limpieza y validación
STEP 2: FEATURE ENGINEERING    # Outliers, split, scaling
STEP 3: MODEL TRAINING         # GridSearchCV con sklearn Pipeline
STEP 4: MODEL EVALUATION       # Métricas y visualizaciones

🎉 PIPELINE COMPLETED SUCCESSFULLY!
📊 Final Results:
  Model: random_forest
  Test AUC-ROC: 0.8196
  Test Accuracy: 0.7464
```

### **Comparar Múltiples Modelos**

```bash
python run_pipeline.py compare
```

**Salida:**
```
🔬 COMPARING 3 MODELS
======================================================================

Model                     AUC-ROC    Accuracy   F1-Score  
--------------------------------------------------------------
Random Forest             0.8196     0.7464     0.8416    
Logistic Regression       0.8186     0.7971     0.8716    
Decision Tree             0.7680     0.7681     0.8431    

🏆 Best model: Random Forest (AUC-ROC: 0.8196)
```

### **Entrenar Modelo Específico**

```bash
# Random Forest
python run_pipeline.py --model-name random_forest

# Logistic Regression
python run_pipeline.py --model-name logistic_regression

# Decision Tree
python run_pipeline.py --model-name decision_tree
```

### **Opciones Avanzadas**

```bash
# Skip data preparation (usar datos ya procesados)
python run_pipeline.py --skip-data-prep --skip-feature-eng

# CV más rápido (3 folds en lugar de 5)
python run_pipeline.py --cv-folds 3

# Sin generar gráficos
python run_pipeline.py --no-plots

# Limpiar archivos generados
python run_pipeline.py clean
```

### **Ejecución por Etapas**

```bash
# 1. Solo limpieza de datos
python -m fase3.dataset

# 2. Solo feature engineering
python -m fase3.features

# 3. Solo entrenamiento
python run_pipeline.py --skip-data-prep --skip-feature-eng
```

---

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/AlejandroDiaz10/MLOps
cd Fase3

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar paquete en modo desarrollo
pip install -e .

# Verificar instalación
python -c "from fase3 import config; print('✅ OK')"
```

---

## 📊 Archivos Generados

Después de ejecutar el pipeline:

**1. Modelos entrenados:**
```
models/
├── random_forest_pipeline.pkl              # Pipeline completo
├── random_forest_pipeline_metadata.json    # Hiperparámetros y métricas
├── logistic_regression_pipeline.pkl
└── decision_tree_pipeline.pkl
```

**2. Visualizaciones:**
```
reports/figures/
├── confusion_matrix_random_forest_(pipeline).png
├── roc_curve_random_forest_(pipeline).png
└── model_comparison.png
```

**3. Datos procesados:**
```
data/processed/
├── X_train.csv                 # Features de entrenamiento (scaled)
├── X_test.csv                  # Features de prueba (scaled)
├── y_train.csv                 # Labels de entrenamiento
├── y_test.csv                  # Labels de prueba
└── test_metrics_pipeline.json  # Métricas finales
```

---

## 🛠️ Stack Tecnológico

**Core:**
- Python 3.12.1
- scikit-learn 1.3.0 (Pipeline, GridSearchCV)
- pandas 2.1.0
- numpy 1.24.3

**Configuración & CLI:**
- pydantic - Validación de configuración
- typer - CLI moderna
- loguru - Logging estructurado

**Visualización:**
- matplotlib 3.7.2
- seaborn 0.12.2

**MLOps (Etapa 4):**
- MLflow (pendiente) - Experiment tracking
- DVC (pendiente) - Data versioning

---

## 📓 Notebooks

| Notebook | Descripción |
|----------|-------------|
| `1.0-data-exploration` | Análisis exploratorio del dataset |
| `2.0-feature-engineering` | Análisis de features y outliers |
| `3.0-model-training` | Comparación de modelos |
| `4.0-sklearn-pipeline` | Demostración de sklearn Pipeline |

**Ejecutar:**
```bash
jupyter notebook notebooks/
```

---

## 🎓 Aprendizajes Clave

**1. Cookiecutter Data Science**
- Estructura estandarizada facilita mantenimiento
- Separación datos/código mejora reproducibilidad
- Organización clara acelera onboarding

**2. Programación Orientada a Objetos**
- Method chaining mejora legibilidad
- Design patterns facilitan extensibilidad
- Separación de responsabilidades reduce bugs

**3. sklearn Pipeline**
- Previene data leakage automáticamente
- Serialización simplificada (un solo objeto)
- Código listo para producción desde el inicio

**4. Experiment Tracking (próximo)**
- Comparación sistemática de modelos
- Versionamiento de artifacts
- Auditoría completa de experimentos

---

## 📄 Licencia

MIT License

---

**Versión:** v3.0.0  
**Team 34** - Machine Learning Course  
**Octubre 2025**