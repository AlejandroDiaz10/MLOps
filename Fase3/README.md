# 🏦 German Credit Risk - ML Pipeline (Fase 3)

Proyecto completo de Machine Learning con MLOps para predicción de riesgo crediticio.

## 📋 Tabla de Contenidos

- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración Inicial](#configuración-inicial)
- [Pipeline Completo](#pipeline-completo)
- [DVC - Versionado de Datos](#dvc---versionado-de-datos)
- [MLflow - Tracking](#mlflow---tracking)
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
│   └── metrics/                  # JSON metrics (tracked by DVC)
├── dvc.yaml                      # DVC pipeline definition
├── params.yaml                   # Hyperparameters
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

# 4. Subir artefactos a S3
dvc push

# 5. Probar API localmente
uvicorn api.main:app --reload

# 6. Test endpoint
curl http://localhost:8000/health

# 7. Build Docker image
docker build -t german-credit-api:1.0.0 .

# 8. Test contenedor
docker run -p 8000:8000 german-credit-api:1.0.0

# 9. Push a DockerHub
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

---

## 👥 Equipo

**Equipo 34**
- Proyecto: German Credit Risk
- Experimento MLflow: `equipo34-german-credit`
- S3 Bucket: `s3://itesm-mna/202502-equipo34/`

---

## 📄 Licencia

Este proyecto es parte del curso de MLOps - ITESM
