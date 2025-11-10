# 🏦 German Credit Risk - ML Pipeline (Fase 3)

Proyecto completo de Machine Learning con MLOps para predicción de riesgo crediticio.

## 📋 Tabla de Contenidos

- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración Inicial](#configuración-inicial)
- [Pipeline Completo](#pipeline-completo)
- [DVC - Versionado de Datos](#dvc---versionado-de-datos)
- [MLflow - Tracking](#mlflow---tracking)
- [Data Drift Monitoring](#-data-drift-monitoring)
- [Testing](#testing)
- [API FastAPI](#api-fastapi)
- [Docker](#docker)
- [Comandos Útiles](#comandos-útiles)

---

## 🗂️ Estructura del Proyecto

```
Fase3/
├── api/                          # FastAPI application
│   ├── main.py                   # API endpoints
│   └── schemas.py                # Pydantic models
├── data/
│   ├── raw/                      # Raw data (tracked by DVC)
│   ├── interim/                  # Cleaned data
│   └── processed/                # Train/test splits (tracked by DVC)
├── fase3/                        # Main package
│   ├── core/                     # Core classes
│   │   ├── data_processor.py    # Data cleaning
│   │   ├── feature_engineer.py  # Feature engineering
│   │   └── model_factory.py     # Model factory
│   ├── modeling/
│   │   ├── train.py             # Training with MLflow
│   │   └── select_best_model.py # Best model selection
│   ├── monitoring/               # Data Drift Monitoring ⭐ NEW
│   │   ├── drift_simulator.py   # Drift scenario generator
│   │   ├── drift_detector.py    # Statistical drift detection
│   │   ├── performance_monitor.py # Performance degradation
│   │   ├── drift_visualizer.py  # Visualizations
│   │   └── monitor_drift.py     # Main orchestrator
│   ├── config.py                 # Configuration
│   ├── dataset.py                # Data preparation
│   ├── features.py               # Feature engineering
│   ├── pipeline_builder.py       # Sklearn pipeline builder
│   └── plots.py                  # Visualization
├── models/                       # Trained models (tracked by DVC)
│   ├── random_forest_pipeline.pkl
│   ├── logistic_regression_pipeline.pkl
│   ├── decision_tree_pipeline.pkl
│   └── best_model_pipeline.pkl   # Best model (for API)
├── reports/
│   ├── figures/                  # Plots
│   │   └── drift/                # Drift monitoring plots ⭐ NEW
│   ├── metrics/                  # JSON metrics (tracked by DVC)
│   └── drift_monitoring_report.json # Drift report ⭐ NEW
├── tests/                        # Test suite
│   ├── conftest.py               # Shared fixtures
│   ├── unit/                     # Unit tests
│   │   ├── test_data_processor.py
│   │   └── test_feature_engineer.py
│   ├── integration/              # Integration tests
│   │   └── test_integration.py
│   └── api/                      # API tests
│       └── test_api.py
├── dvc.yaml                      # DVC pipeline definition
├── params.yaml                   # Hyperparameters
├── pytest.ini                    # Pytest configuration
├── Dockerfile                    # Container definition
└── requirements.txt              # Python dependencies
```

---

## ⚙️ Configuración Inicial

### 1. Instalación de dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar AWS (para DVC)

```bash
# Configurar perfil AWS
aws configure --profile equipo34

# Verificar acceso a S3
aws s3 ls s3://itesm-mna/202502-equipo34/ --profile equipo34
```

### 3. Configurar DVC

```bash
# Inicializar DVC (ya está hecho)
dvc init

# Configurar remote S3
dvc remote add -d team_remote s3://itesm-mna/202502-equipo34
dvc remote modify team_remote profile equipo34

# Verificar configuración
dvc remote list
```

### 4. Verificar MLflow

```bash
# Probar conexión al servidor MLflow
python mlflow_setup.py

# Servidor MLflow: http://34.67.152.248:5000/
```

---

## 🚀 Pipeline Completo

### Opción 1: Ejecutar TODO el pipeline con DVC

```bash
# Ejecutar TODO el pipeline (preprocesamiento + entrenamiento de 3 modelos + selección)
dvc repro

# Esto ejecutará:
# 1. prepare_data
# 2. feature_engineering
# 3. train_random_forest
# 4. train_logistic_regression
# 5. train_decision_tree
# 6. select_best_model
```

### Opción 2: Ejecutar paso por paso

```bash
# 1. Preparación de datos
python -m fase3.dataset

# 2. Feature engineering
python -m fase3.features

# 3. Entrenar modelos (uno por uno)
python -m fase3.modeling.train --model-name random_forest
python -m fase3.modeling.train --model-name logistic_regression
python -m fase3.modeling.train --model-name decision_tree

# 4. Seleccionar mejor modelo
python -m fase3.modeling.select_best_model
```

### Opción 3: Usar run_pipeline.py (alternativa)

```bash
# Entrenar y comparar todos los modelos
python run_pipeline.py compare

# Solo entrenar un modelo
python run_pipeline.py --model-name random_forest
```

---

## 📊 DVC - Versionado de Datos

### ¿Qué versiona DVC?

DVC versiona:
- **Datos raw**: `data/raw/german_credit_modified.csv`
- **Datos procesados**: `data/processed/*.csv`
- **Modelos entrenados**: `models/*.pkl`
- **Métricas**: `reports/metrics/*.json`

### Comandos DVC importantes

```bash
# Ver estado del pipeline
dvc status

# Ver DAG del pipeline
dvc dag

# Push a S3 (subir datos y modelos)
dvc push

# Pull desde S3 (descargar datos y modelos)
dvc pull

# Ver métricas
dvc metrics show

# Comparar métricas entre runs
dvc metrics diff
```

### Estructura de dvc.yaml

El archivo `dvc.yaml` define 6 stages:

1. **prepare_data**: Limpieza de datos
2. **feature_engineering**: Train/test split + scaling
3. **train_random_forest**: Entrenar Random Forest
4. **train_logistic_regression**: Entrenar Logistic Regression
5. **train_decision_tree**: Entrenar Decision Tree
6. **select_best_model**: Seleccionar mejor modelo basado en AUC-ROC

### ¿Qué son los hashes?

Los hashes MD5 en `dvc.lock` identifican la versión exacta de cada archivo:
- Detectan si cambiaron los datos
- Permiten reproducibilidad exacta
- Se usan para cache inteligente

```bash
# Ver qué archivos cambiaron
dvc status

# Output ejemplo:
# data/processed/X_train.csv:
#   hash: b3e75a54904bd4152e637cb5c50d58a3
#   changed: False
```

---

## 🔬 MLflow - Tracking

### Servidor MLflow

```
URL: http://34.67.152.248:5000/
Experimento: equipo34-german-credit
```

### ¿Qué se registra en MLflow?

Para CADA modelo entrenado:

**Parámetros:**
- model_name
- cv_folds
- test_size
- random_state
- Hiperparámetros del modelo (best_params de GridSearch)

**Métricas:**
- `cv_best_score`: Mejor score en cross-validation
- `test_accuracy`: Accuracy en test set
- `test_precision`: Precision en test set
- `test_recall`: Recall en test set
- `test_f1_score`: F1-Score en test set
- `test_auc_roc`: AUC-ROC en test set ⭐
- `training_time_seconds`: Tiempo de entrenamiento

**Artefactos:**
- Pipeline completo (.pkl)
- Metadata (.json)

**Modelos registrados:**
- `equipo34-german-credit_random_forest`
- `equipo34-german-credit_logistic_regression`
- `equipo34-german-credit_decision_tree`
- `equipo34-german-credit_production` ⭐ (mejor modelo)

### Ver experimentos

```bash
# En navegador
http://34.67.152.248:5000/

# O programáticamente
python -c "
import mlflow
mlflow.set_tracking_uri('http://34.67.152.248:5000')
runs = mlflow.search_runs(experiment_names=['equipo34-german-credit'])
print(runs[['metrics.test_auc_roc', 'params.model_name']])
"
```

---

## 📉 Data Drift Monitoring

### Sistema de Monitoreo de Drift

Sistema completo para detectar cambios en la distribución de datos y degradación del modelo en producción.

### Ejecutar monitoreo completo

```bash
# Comando principal (ejecuta todo el workflow)
python3 -m fase3.monitoring.monitor_drift
```

### ¿Qué hace el monitoreo?

El sistema ejecuta automáticamente:

1. **Carga datos de referencia** (test set) y modelo entrenado
2. **Genera 9 escenarios de drift**:
   - `baseline`: Sin drift (control)
   - `mild_mean_shift`: Desplazamiento ligero de medias (20%)
   - `moderate_mean_shift`: Desplazamiento moderado (50%)
   - `severe_mean_shift`: Desplazamiento severo (80%)
   - `mild_variance_change`: Cambio en varianza (20%)
   - `moderate_outliers`: Introducción de outliers (10%)
   - `severe_missing`: Datos faltantes (30%)
   - `mild_concept_drift`: Cambio X→y ligero (10%)
   - `severe_concept_drift`: Cambio X→y severo (30%)

3. **Detecta drift** usando tests estadísticos:
   - **Kolmogorov-Smirnov (KS)**: p-value < 0.05
   - **Population Stability Index (PSI)**: threshold 0.1
   - **Jensen-Shannon Divergence**: distancia [0,1]

4. **Monitorea performance**:
   - Compara 5 métricas vs baseline
   - Detecta degradación > 5%
   - Genera alertas automáticas

5. **Crea visualizaciones**:
   - Dashboard completo
   - Drift heatmap
   - Comparación de distribuciones
   - Gráficos de performance

6. **Genera reporte JSON** con resultados estructurados

### Outputs generados

```bash
reports/
├── figures/drift/
│   ├── drift_dashboard.png              # Panel completo de monitoreo
│   ├── drift_heatmap_moderate.png       # Matriz de drift scores
│   ├── feature_distributions_severe.png # Histogramas comparativos
│   └── performance_comparison.png       # Métricas por escenario
└── drift_monitoring_report.json         # Reporte estructurado
```

### Ver resultados

```bash
# Ver dashboard principal
open reports/figures/drift/drift_dashboard.png  # macOS
xdg-open reports/figures/drift/drift_dashboard.png  # Linux

# Ver reporte JSON
cat reports/drift_monitoring_report.json | jq .

# Ver resumen de drift
cat reports/drift_monitoring_report.json | jq '.summary'
```

### Interpretación de resultados

**Niveles de Drift:**
- **None** (0%): Sin cambios detectados ✅
- **Minor** (1-25%): Cambios menores, monitorear ⚠️
- **Moderate** (26-50%): Revisión recomendada 🔄
- **Severe** (51%+): Acción urgente requerida 🚨

**Niveles de Degradación (AUC-ROC):**
- **< 5%**: Rendimiento aceptable ✅
- **5-10%**: Degradación moderada, investigar 🔄
- **> 10%**: Degradación severa, retrain urgente 🚨

**Acciones recomendadas:**
- **No action**: Continuar monitoreo normal
- **Monitor closely**: Aumentar frecuencia de monitoreo
- **Review pipeline**: Investigar causas y ajustar feature engineering
- **Retrain immediately**: Reentrenar modelo con datos recientes

### Opciones avanzadas

```bash
# Ver ayuda
python3 -m fase3.monitoring.monitor_drift --help

# Ejecutar sin visualizaciones (solo detección)
python3 -m fase3.monitoring.monitor_drift --no-create-visualizations

# Solo generar escenarios (sin análisis)
python3 -m fase3.monitoring.monitor_drift --no-detect-drift --no-monitor-performance
```

### Componentes del sistema

**Módulos principales:**
- `drift_simulator.py`: Genera datos con diferentes tipos de drift
- `drift_detector.py`: Tests estadísticos (KS, PSI, JS)
- `performance_monitor.py`: Compara métricas vs baseline
- `drift_visualizer.py`: Crea gráficos profesionales
- `monitor_drift.py`: Orquesta el workflow completo

**Tests estadísticos:**
1. **Kolmogorov-Smirnov**: Mide máxima diferencia entre distribuciones acumulativas
2. **PSI (Population Stability Index)**: Estándar en industria financiera (< 0.1 = estable)
3. **Jensen-Shannon**: Distancia simétrica entre distribuciones [0,1]

### Integración con workflow MLOps

```bash
# Workflow recomendado:

# 1. Entrenar modelo
dvc repro

# 2. Monitorear drift
python3 -m fase3.monitoring.monitor_drift

# 3. Revisar resultados
open reports/figures/drift/drift_dashboard.png

# 4. Si drift severo detectado → retrain
dvc repro

# 5. Repetir monitoreo
python3 -m fase3.monitoring.monitor_drift
```

### Frecuencia recomendada

- **Desarrollo**: Después de cada entrenamiento
- **Staging**: Semanalmente con datos de staging
- **Producción**: Diariamente o semanalmente dependiendo del volumen

---

## 🧪 Testing

### Suite de Tests Completa

El proyecto incluye **59 tests automatizados** que cubren:
- ✅ **25+ tests unitarios**: Componentes individuales
- ✅ **10+ tests de integración**: Pipeline end-to-end
- ✅ **15+ tests de API**: Endpoints FastAPI
- ✅ **85% coverage**: Módulos core y API

### Ejecutar todos los tests

```bash
# Comando principal (con coverage)
pytest -v

# Output esperado:
# ===== 59 passed in 6.08s =====
# Coverage: 84.51%
```

### Ejecutar tests por categoría

```bash
# Solo tests unitarios (rápidos)
pytest -v -m unit

# Solo tests de integración
pytest -v -m integration

# Solo tests de API
pytest -v -m api

# Excluir tests lentos
pytest -v -m "not slow"
```

### Ver cobertura de código

```bash
# Coverage en terminal
pytest -v --cov=fase3.core --cov=api --cov-report=term-missing

# Generar reporte HTML
pytest -v --cov=fase3.core --cov=api --cov-report=html

# Abrir reporte en navegador
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### ¿Qué se está testeando?

**Tests Unitarios** (`tests/unit/`)
- `test_data_processor.py` (16 tests)
  - Carga de datos, limpieza, validación
  - Traducción de columnas, manejo de NaNs
  - Validación de rangos y tipos de datos
  
- `test_feature_engineer.py` (16 tests)
  - Train-test split, detección de outliers
  - Feature scaling, guardado de datos
  - Manejo de errores y edge cases

**Tests de Integración** (`tests/integration/`)
- `test_integration.py` (10 tests)
  - Pipeline completo: raw → processed → predictions
  - Entrenamiento con diferentes modelos
  - Serialización y carga de modelos
  - Reproducibilidad y performance mínima

**Tests de API** (`tests/api/`)
- `test_api.py` (17 tests)
  - Endpoints: health, model-info, predict
  - Validación de inputs (Pydantic schemas)
  - Manejo de errores y casos edge
  - Consistencia de predicciones

### Estructura de fixtures

Los fixtures reutilizables en `conftest.py` incluyen:
- `sample_raw_data`: 100 muestras sintéticas
- `sample_clean_data`: Datos limpios sin NaNs
- `sample_train_test_split`: Split precomputado
- `api_client`: Cliente TestClient para FastAPI
- `temp_dirs`: Directorios temporales para tests

### Comandos útiles adicionales

```bash
# Ejecutar test específico
pytest tests/unit/test_data_processor.py::TestDataProcessor::test_load_data -v

# Detener en primer fallo
pytest -x

# Mostrar prints en tests
pytest -v -s

# Ejecutar en paralelo (requiere pytest-xdist)
pytest -v -n auto

# Ver resumen de tests
pytest --collect-only
```

### Configuración (pytest.ini)

El archivo `pytest.ini` configura:
- **Descubrimiento**: Solo archivos `test_*.py` en `tests/`
- **Markers**: `unit`, `integration`, `api`, `slow`
- **Coverage**: Mide `fase3.core` y `api` (threshold: 70%)
- **Output**: Verbose, traceback corto, warnings deshabilitados

---

## 🌐 API FastAPI

### Iniciar API localmente

```bash
# Método 1: Directamente
uvicorn api.main:app --reload --port 8000

# Método 2: Con configuración
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints disponibles

```bash
# 1. Root
curl http://localhost:8000/

# 2. Health check
curl http://localhost:8000/health

# 3. Model info
curl http://localhost:8000/model-info

# 4. Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "checking_account": 1,
    "duration": 24,
    "credit_history": 2,
    "purpose": 3,
    "amount": 5000,
    "savings_account": 1,
    "employment_duration": 3,
    "installment_rate": 2,
    "personal_status": 2,
    "other_debtors": 1,
    "residence_duration": 3,
    "property": 2,
    "age": 35,
    "other_installment_plans": 1,
    "housing": 2,
    "existing_credits": 1,
    "job": 2,
    "dependents": 1,
    "telephone": 1,
    "foreign_worker": 1
  }'
```

### Documentación interactiva

```
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
```

---

## 🐳 Docker

### Construir imagen

```bash
# Build
docker build -t german-credit-api:latest .

# Build con tag versionado
docker build -t german-credit-api:1.0.0 .
```

### Ejecutar contenedor

```bash
# Run
docker run -p 8000:8000 german-credit-api:latest

# Run en background
docker run -d -p 8000:8000 --name credit-api german-credit-api:latest

# Ver logs
docker logs credit-api

# Detener
docker stop credit-api
```

### Publicar en DockerHub

```bash
# Login
docker login

# Tag para DockerHub
docker tag german-credit-api:latest tuusuario/german-credit-api:1.0.0
docker tag german-credit-api:latest tuusuario/german-credit-api:latest

# Push
docker push tuusuario/german-credit-api:1.0.0
docker push tuusuario/german-credit-api:latest
```

### Usar desde DockerHub

```bash
# Pull
docker pull tuusuario/german-credit-api:latest

# Run
docker run -p 8000:8000 tuusuario/german-credit-api:latest
```

---

## 📈 Flujo de Trabajo Completo

### Workflow típico:

```bash
# 1. Entrenar todos los modelos
dvc repro

# 2. Verificar en MLflow que se registraron
# Ir a http://34.67.152.248:5000/

# 3. Verificar que se seleccionó best_model
ls -lh models/best_model_pipeline.pkl

# 4. Monitorear drift ⭐ NEW
python3 -m fase3.monitoring.monitor_drift

# 5. Revisar resultados de drift
open reports/figures/drift/drift_dashboard.png
cat reports/drift_monitoring_report.json

# 6. Ejecutar tests
pytest -v

# 7. Subir artefactos a S3
dvc push

# 8. Probar API localmente
uvicorn api.main:app --reload

# 9. Test endpoint
curl http://localhost:8000/health

# 10. Build Docker image
docker build -t german-credit-api:1.0.0 .

# 11. Test contenedor
docker run -p 8000:8000 german-credit-api:1.0.0

# 12. Push a DockerHub
docker tag german-credit-api:1.0.0 tuusuario/german-credit-api:1.0.0
docker push tuusuario/german-credit-api:1.0.0
```

---

## 📝 Notas Importantes

### Sobre DVC y Git

- **Git** versiona: código, configuración, dvc.yaml, dvc.lock
- **DVC** versiona: datos, modelos, métricas
- Los `.gitignore` en `data/*/` evitan subir archivos pesados a Git

### Sobre el best_model

- Se selecciona automáticamente basado en `test_auc_roc`
- Se copia a `models/best_model_pipeline.pkl`
- Es el que usa la API por defecto
- Se registra en MLflow como `*_production`

### Sobre las métricas

Cada modelo tiene 2 archivos JSON:
1. `reports/metrics/{model}_metrics.json` → Para DVC
2. `models/{model}_pipeline_metadata.json` → Para API

### Sobre los tests

- Los tests garantizan estabilidad del código
- Coverage de 85% en módulos core
- Tests de integración validan pipeline completo
- Tests de API aseguran endpoints funcionales

### Sobre el drift monitoring

- Ejecutar después de cada entrenamiento en desarrollo
- Ejecutar periódicamente en producción (diario/semanal)
- AUC-ROC es la métrica más sensible para detectar impacto de drift
- Concept drift (cambio X→y) degrada más que feature drift
- Usar visualizaciones para comunicar drift a stakeholders

---

## 👥 Equipo

**Equipo 34**
- Proyecto: German Credit Risk
- Experimento MLflow: `equipo34-german-credit`
- S3 Bucket: `s3://itesm-mna/202502-equipo34/`

---

## 📄 Licencia

Este proyecto es parte del curso de MLOps - ITESM