# Fase 2: German Credit Risk Prediction

[![Python 3.12.1](https://img.shields.io/badge/python-3.12.1-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyecto académico de Machine Learning para predecir riesgo crediticio utilizando el dataset German Credit, estructurado siguiendo mejores prácticas de MLOps con Cookiecutter Data Science y programación orientada a objetos.

**Equipo:** Team 34  
**Curso:** Machine Learning  
**Fecha:** Octubre 2025

---

## 📋 Tabla de Contenidos

- [Estado del Proyecto](#estado-del-proyecto)
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Etapas Completadas](#etapas-completadas)
  - [Etapa 1: Estructura del Proyecto](#etapa-1-estructura-del-proyecto)
  - [Etapa 2: Refactorización OOP](#etapa-2-refactorización-oop)
  - [Etapa 3: sklearn Pipeline](#etapa-3-sklearn-pipeline-best-practices)
- [Próximos Pasos](#próximos-pasos-etapa-4)
- [Guías Detalladas](#guías-detalladas)
- [Tecnologías](#tecnologías)

---

## 🎯 Estado del Proyecto

| Etapa | Estado | Descripción |
|-------|--------|-------------|
| **1. Estructura** | ✅ Completa | Cookiecutter Data Science + Setup inicial |
| **2. Refactorización OOP** | ✅ Completa | Código modular con clases y design patterns |
| **3. sklearn Pipeline** | ✅ Completa | Pipeline automatizado con best practices |
| **4. MLflow/DVC** | ⏳ Pendiente | Tracking de experimentos y versionamiento |

**Versión actual:** `v3.0.0`

---

## 🎯 Descripción del Proyecto

Sistema de predicción de riesgo crediticio que clasifica clientes como "buen crédito" (1) o "mal crédito" (0) utilizando el dataset **German Credit** de UCI.

**Características implementadas:**
- ✅ Arquitectura orientada a objetos (OOP)
- ✅ Pipeline de ML completo y modular
- ✅ Sklearn Pipeline con GridSearchCV
- ✅ Configuración validada con Pydantic
- ✅ Method chaining para API limpia
- ✅ Exception handling robusto
- ✅ Visualizaciones automáticas
- ✅ Backward compatibility

**Métrica objetivo:** AUC-ROC ≥ 0.75

---

## 🏗️ Arquitectura

### **Arquitectura en Capas**
```
┌─────────────────────────────────────────────┐
│  CAPA 1: CLI / Scripts                      │  ← run_pipeline.py, dataset.py
├─────────────────────────────────────────────┤
│  CAPA 2: Pipeline Orchestrator              │  ← MLPipeline (facade)
├─────────────────────────────────────────────┤
│  CAPA 3: Core Business Logic (OOP)          │  ← DataProcessor, FeatureEngineer, ModelTrainer, PipelineBuilder
│         + sklearn Pipeline                  │     
├─────────────────────────────────────────────┤
│  CAPA 4: Artifacts & Outputs                │  ← models/, reports/, data/
└─────────────────────────────────────────────┘
```

### **Flujo de Datos**
```mermaid
graph LR
    A[Raw Data] --> B[DataProcessor]
    B --> C[Cleaned Data]
    C --> D[FeatureEngineer]
    D --> E[Train/Test Split]
    E --> F[sklearn Pipeline]
    F --> G[Trained Model]
    G --> H[Predictions]
    H --> I[Metrics & Plots]
```

### **Componentes Principales**

| Componente | Responsabilidad | Patrón |
|------------|----------------|--------|
| `DataProcessor` | Limpieza y validación | Method Chaining |
| `FeatureEngineer` | Feature engineering | Method Chaining |
| `ModelTrainer` | Entrenamiento | Template Method |
| `ModelEvaluator` | Evaluación | - |
| `ModelFactory` | Creación de modelos | Factory Pattern |
| `PipelineBuilder` | Construcción de sklearn Pipeline | Builder Pattern |
| `MLPipeline` | Orquestación | Facade Pattern |

---

## 📁 Estructura del Proyecto
```
Fase2/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias
├── pyproject.toml              # Configuración del proyecto
├── run_pipeline.py             # 🎯 Script principal (EMPEZAR AQUÍ)
│
├── data/
│   ├── raw/                    # Datos originales (inmutables)
│   │   └── german_credit_modified.csv
│   ├── interim/                # Datos limpiados
│   │   └── german_credit_cleaned.csv
│   ├── processed/              # Datos finales para ML
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── external/
│
├── models/                     # Modelos entrenados (.pkl)
│   ├── random_forest_pipeline.pkl
│   ├── random_forest_pipeline_metadata.json
│   └── ...
│
├── notebooks/                  # Análisis interactivo
│   ├── 1.0-t34-data-exploration.ipynb
│   ├── 2.0-t34-feature-engineering.ipynb
│   ├── 3.0-t34-model-training.ipynb
│   └── 4.0-t34-sklearn-pipeline-best-practices.ipynb  
│
├── reports/
│   └── figures/                # Visualizaciones
│       ├── confusion_matrix_*.png
│       ├── roc_curve_*.png
│       └── model_comparison.png
│
├── docs/                       # Documentación adicional
│   └── SKLEARN_PIPELINE_GUIDE.md
│
└── fase2/                      # 📦 Código fuente (paquete Python)
    ├── __init__.py
    ├── config.py               # Configuración con Pydantic
    ├── exceptions.py           # Custom exceptions
    ├── pipeline.py             # MLPipeline orchestrator
    ├── plots.py                # Funciones de visualización
    ├── transformers.py         # Custom sklearn transformers
    ├── pipeline_builder.py     # Construcción de sklearn Pipeline
    ├── demo_pipeline.py        # Script de demostración
    │
    ├── core/                   # Lógica de negocio (OOP)
    │   ├── data_processor.py
    │   ├── feature_engineer.py
    │   ├── model_factory.py
    │   ├── trainer.py
    │   └── evaluator.py
    │
    ├── utils/                  # Utilidades
    │   ├── logger.py
    │   └── validators.py
    │
    ├── dataset.py              # CLI para limpieza
    ├── features.py             # CLI para features
    └── modeling/
        ├── train.py            # CLI para entrenamiento
        └── predict.py          # CLI para predicción
```

---

## 🚀 Instalación

### **1. Prerrequisitos**

- Python 3.12.1+
- pip
- virtualenv (recomendado)
- Git

### **2. Setup Completo**
```bash
# Clonar repositorio
git clone https://github.com/AlejandroDiaz10/MLOps
cd Fase2

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar paquete en modo desarrollo
pip install -e .

# Verificar instalación
python -c "from fase2 import config; print('✅ Installation successful')"
```

### **3. Verificar Estructura**
```bash
# Ver estructura de directorios
tree -L 2  # Linux/Mac
# o
ls -R      # Alternativa

# Verificar que existen los datos raw
ls data/raw/german_credit_modified.csv
```

---

## ⚡ Uso Rápido

### **Opción 1: Pipeline Completo con sklearn (Recomendado)**
```bash
# Ejecutar TODO el pipeline de ML
python run_pipeline.py sklearn
```

**Esto ejecuta automáticamente:**
1. ✅ Data preparation → `data/interim/german_credit_cleaned.csv`
2. ✅ Feature engineering → `data/processed/X_train.csv`, etc.
3. ✅ sklearn Pipeline training con GridSearchCV
4. ✅ Evaluación en test set
5. ✅ Generación de visualizaciones

**Archivos generados:**
```
models/
└── random_forest_pipeline.pkl              # Pipeline completo
└── random_forest_pipeline_metadata.json    # Metadata

data/processed/
└── test_predictions_pipeline.csv
└── test_metrics_pipeline.json

reports/figures/
└── confusion_matrix_random_forest_(pipeline).png
└── roc_curve_random_forest_(pipeline).png
```

### **Opción 2: Demo Rápido**
```bash
# Demostración del sklearn Pipeline 
python -m fase2.demo_pipeline
```

### **Opción 3: Paso por Paso**
```bash
# 1. Limpiar datos
python -m fase2.dataset

# 2. Feature engineering
python -m fase2.features

# 3. Entrenar modelo
python run_pipeline.py sklearn

# 4. Diferentes modelos
python run_pipeline.py sklearn --model-name logistic_regression
python run_pipeline.py sklearn --model-name decision_tree
```

---

## ✅ Etapas Completadas

### **Etapa 1: Estructura del Proyecto**

**Objetivo:** Establecer estructura estandarizada del proyecto.

**Implementado:**
- ✅ Cookiecutter Data Science structure
- ✅ Configuración con Pydantic (`config.py`)
- ✅ Setup de logging (`utils/logger.py`)
- ✅ Git repository inicializado

**Documentación:** Ver estructura en [Estructura del Proyecto](#estructura-del-proyecto)

---

### **Etapa 2: Refactorización OOP**

**Objetivo:** Organizar código en módulos con responsabilidades claras usando OOP.

#### **Clases Implementadas**

**1. DataProcessor (`core/data_processor.py`)**
- Carga de datos raw
- Traducción de columnas
- Limpieza de whitespace
- Conversión a tipos numéricos
- Validación de target
- Manejo de missing values
- Validación de rangos categóricos
- Remoción de duplicados

**Uso:**
```python
from fase2.core.data_processor import DataProcessor

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
```

**2. FeatureEngineer (`core/feature_engineer.py`)**
- Detección de outliers
- Separación de features/target
- Train-test split
- Feature scaling
- Guardado de artefactos

**Uso:**
```python
from fase2.core.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
engineer \
    .load_data() \
    .detect_outliers() \
    .split_target() \
    .train_test_split() \
    .scale_features() \
    .save_all()
```

**3. ModelTrainer (`core/trainer.py`)**
- Entrenamiento con GridSearchCV
- Cross-validation
- Búsqueda de hiperparámetros
- Guardado de modelos

**4. ModelEvaluator (`core/evaluator.py`)**
- Predicción en test set
- Cálculo de métricas
- Classification report
- Guardado de resultados

**5. ModelFactory (`core/model_factory.py`)**
- Factory Pattern para crear modelos
- Grids de hiperparámetros predefinidos
- Modelos soportados: Random Forest, Logistic Regression, Decision Tree

**6. MLPipeline (`pipeline.py`)**
- Orquestador maestro
- Ejecuta workflow completo
- Comparación de múltiples modelos

#### **Design Patterns Aplicados**

| Pattern | Dónde | Para Qué |
|---------|-------|----------|
| **Factory** | `ModelFactory` | Creación de modelos |
| **Builder** | `PipelineBuilder` | Construcción de pipelines |
| **Template Method** | `MLPipeline` | Workflow estándar |
| **Facade** | `MLPipeline` | Interfaz simplificada |
| **Method Chaining** | Todas las clases core | API fluida |

#### **Principios SOLID**

- ✅ **Single Responsibility:** Cada clase tiene una responsabilidad
- ✅ **Open/Closed:** Extensible vía herencia/composición
- ✅ **Dependency Inversion:** Config inyectable

---

### **Etapa 3: sklearn Pipeline (Best Practices)**

**Objetivo:** Implementar pipeline de scikit-learn que automatice preprocesamiento, entrenamiento y evaluación.

#### **¿Por qué sklearn Pipeline?**

**Problemas sin Pipeline:**
- ❌ Data leakage (fit en todo el dataset)
- ❌ Múltiples objetos a serializar
- ❌ Fácil olvidar preprocessing en producción
- ❌ Difícil reproducir experimentos

**Solución con Pipeline:**
- ✅ Un solo objeto `.pkl` serializable
- ✅ Preprocessing automático en train y test
- ✅ No data leakage (fit solo en train)
- ✅ GridSearchCV sobre todo el pipeline
- ✅ Listo para producción

#### **Componentes Implementados**

**1. Custom Transformers (`transformers.py`)**

Transformers compatibles con sklearn que siguen la API `fit/transform`:
```python
class OutlierRemover(BaseEstimator, TransformerMixin):
    """Clip outliers usando IQR method."""
    def fit(self, X, y=None):
        # Aprender bounds del train set
        return self
    
    def transform(self, X):
        # Aplicar clipping
        return X_clipped
```

**Transformers disponibles:**
- `OutlierRemover` - Clip outliers con IQR
- `TypeConverter` - Conversión a tipos numéricos
- `CategoricalValidator` - Validación de categorías
- `DataFrameSelector` - Selección de columnas

**2. PipelineBuilder (`pipeline_builder.py`)**

Constructor de pipelines sklearn:
```python
from fase2.pipeline_builder import PipelineBuilder

builder = PipelineBuilder()

# Pipeline simple
pipeline = builder.build_pipeline('random_forest')

# Pipeline con GridSearch
grid_pipeline = builder.build_grid_search_pipeline(
    model_name='random_forest',
    cv_folds=5
)
```

**Estructura del Pipeline:**
```
sklearn.pipeline.Pipeline
├── imputer (SimpleImputer)        # Imputación de NaN
├── scaler (StandardScaler)        # Escalado de features
└── model (RandomForestClassifier) # Modelo ML
```

#### **Flujo de Trabajo con sklearn Pipeline**
```
1. Data Cleaning (dataset.py)
   └── Limpieza manual → interim/

2. Feature Engineering (features.py)
   └── Outliers, train-test split → processed/

3. sklearn Pipeline (pipeline_builder.py)
   ├── Imputation (aprende de train)
   ├── Scaling (aprende de train)
   └── Model training
```

#### **Uso del sklearn Pipeline**

**Método 1: CLI (Recomendado)**
```bash
python run_pipeline.py sklearn
```

**Método 2: Programático**
```python
from fase2.pipeline_builder import PipelineBuilder

builder = PipelineBuilder()
pipeline = builder.build_grid_search_pipeline('random_forest')
pipeline.fit(X_train, y_train)

# Guardar
import joblib
joblib.dump(pipeline.best_estimator_, 'model.pkl')

# Cargar y usar
loaded_pipeline = joblib.load('model.pkl')
predictions = loaded_pipeline.predict(X_new)
```

**Método 3: Notebook Interactivo**
```bash
jupyter notebook notebooks/4.0-t34-sklearn-pipeline-best-practices.ipynb
```

#### **Best Practices Implementadas**

| Practice | Implementación | Beneficio |
|----------|----------------|-----------|
| **No Data Leakage** | `fit()` solo en train | Resultados válidos |
| **Single Object** | Pipeline completo en .pkl | Fácil deployment |
| **Grid Search** | Sobre todo el pipeline | Optimal hyperparams |
| **Reproducibilidad** | `random_state=42` everywhere | Resultados consistentes |
| **Documentación** | Docstrings + Type hints | Código mantenible |

#### **Archivos Clave de la Etapa 3**
```
fase2/
├── transformers.py          # Custom sklearn transformers
├── pipeline_builder.py      # Constructor de pipelines
└── demo_pipeline.py         # Script de demostración

notebooks/
└── 4.0-t34-sklearn-pipeline-best-practices.ipynb

docs/
└── SKLEARN_PIPELINE_GUIDE.md  # Guía completa
```

---

## 🔜 Próximos Pasos (Etapa 4)

### **Objetivos de la Etapa 4**

Implementar tracking de experimentos y versionamiento de datos/modelos.

#### **Tareas Pendientes**

**1. MLflow - Tracking de Experimentos**
- [ ] Setup de MLflow server
- [ ] Logging automático de parámetros
- [ ] Logging de métricas (AUC, accuracy, etc.)
- [ ] Registro de artifacts (modelos, plots)
- [ ] Comparación de runs en UI
- [ ] Registro de modelos en Model Registry

**2. DVC - Versionamiento de Datos**
- [ ] Inicializar DVC en el proyecto
- [ ] Configurar remote storage
- [ ] Versionar `data/raw/`
- [ ] Versionar `data/processed/`
- [ ] Versionar `models/`
- [ ] Pipeline de DVC

**3. Visualización de Resultados**
- [ ] Dashboard de MLflow
- [ ] Comparación de experimentos
- [ ] Gráficos de convergencia
- [ ] Análisis de hiperparámetros

**4. Gestión de Modelos**
- [ ] Model Registry con versiones
- [ ] Metadata completo por versión
- [ ] Transición de stages (Staging → Production)
- [ ] Rollback capability

#### **Recursos para Etapa 4**

**Documentación:**
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [DVC Get Started](https://dvc.org/doc/start)
- [DVC with Google Drive](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)

---

### **Notebooks Disponibles**

1. **`1.0-t34-data-exploration.ipynb`** - EDA del dataset
2. **`2.0-t34-feature-engineering.ipynb`** - Exploración de features
3. **`3.0-t34-model-training.ipynb`** - Entrenamiento de modelos
4. **`4.0-t34-sklearn-pipeline-best-practices.ipynb`** - Demo de sklearn Pipeline

### **Scripts Útiles**

| Script | Propósito | Comando |
|--------|-----------|---------|
| `run_pipeline.py` | Pipeline completo | `python run_pipeline.py sklearn` |
| `fase2/demo_pipeline.py` | Demo rápido | `python -m fase2.demo_pipeline` |
| `fase2/dataset.py` | Solo limpieza | `python -m fase2.dataset` |
| `fase2/features.py` | Solo features | `python -m fase2.features` |

---

## 🛠️ Tecnologías

**Core:**
- Python 3.12.1
- pandas 2.1.0
- numpy 1.24.3
- scikit-learn 1.3.0

**Configuración y Validación:**
- pydantic - Validación de configuración
- loguru - Logging estructurado
- typer - CLI moderna

**Visualización:**
- matplotlib 3.7.2
- seaborn 0.12.2

**ML Pipeline:**
- scikit-learn Pipeline
- GridSearchCV
- Custom Transformers

**Desarrollo:**
- Cookiecutter Data Science
- Git
- Jupyter
- joblib

**Etapa 4:**
- MLflow - Experiment tracking
- DVC - Data versioning
- (Opcional) GitHub Actions - CI/CD

---

## 🐛 Troubleshooting

### **Problema: Datos no encontrados**
```bash
# Verificar estructura
ls data/raw/
ls data/processed/

# Si faltan, ejecutar:
python -m fase2.dataset
python -m fase2.features
```

### **Problema: Módulo no encontrado**
```bash
# Reinstalar paquete
pip install -e .
```

### **Problema: GridSearchCV muy lento**
Para acelerar durante desarrollo, edita `fase2/core/model_factory.py`:
```python
# Grid reducido para testing
'random_forest': {
    'n_estimators': [100],      # Solo 1 valor
    'max_depth': [20],          # Solo 1 valor
    'min_samples_split': [2]    # Solo 1 valor
}
```

---

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

---

**Última actualización:** Octubre 2025  
**Versión:** v3.0.0  