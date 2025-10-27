# Fase 2: German Credit Risk Prediction

[![Python 3.12.1](https://img.shields.io/badge/python-3.12.1-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyecto académico de Machine Learning para predecir riesgo crediticio utilizando el dataset German Credit, estructurado siguiendo mejores prácticas de MLOps con Cookiecutter Data Science.

**Equipo:** Team 34  
**Curso:** Machine Learning  
**Fecha:** Octubre 2025

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Pipeline Completo](#pipeline-completo)
- [Resultados](#resultados)
- [Tecnologías](#tecnologías)
- [Contribuidores](#contribuidores)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema de predicción de riesgo crediticio utilizando el dataset **German Credit**. El objetivo es clasificar clientes como "buen crédito" (1) o "mal crédito" (0) basándose en características demográficas y financieras.

**Características principales:**
- ✅ Pipeline de ML completo y modular
- ✅ Estructura estandarizada con Cookiecutter Data Science
- ✅ Código reutilizable y mantenible
- ✅ Versionamiento de modelos y experimentos
- ✅ Documentación completa

**Métrica objetivo:** AUC-ROC ≥ 0.75

---

## 📁 Estructura del Proyecto
```
Fase2/
├── LICENSE              # Licencia MIT
├── Makefile             # Comandos de automatización
├── README.md            # Este archivo
├── pyproject.toml       # Configuración del proyecto
├── requirements.txt     # Dependencias Python
│
├── data/
│   ├── raw/             # Datos originales (inmutables)
│   ├── interim/         # Datos intermedios (limpiados)
│   ├── processed/       # Datos finales para modelado
│   └── external/        # Datos de terceros
│
├── docs/                # Documentación con mkdocs
│
├── models/              # Modelos entrenados (.pkl) y metadata
│
├── notebooks/           # Jupyter notebooks de exploración
│   ├── 1.0-t34-data-exploration.ipynb
│   ├── 2.0-t34-feature-engineering.ipynb
│   └── 3.0-t34-model-training.ipynb
│
├── references/          # Diccionarios de datos, manuales
│
├── reports/             # Análisis generados (HTML, PDF)
│   └── figures/         # Gráficos y visualizaciones
│
└── fase2/               # Código fuente del proyecto
    ├── __init__.py
    ├── config.py        # Configuración global
    ├── dataset.py       # Carga y limpieza de datos
    ├── features.py      # Feature engineering
    ├── plots.py         # Visualizaciones
    └── modeling/
        ├── train.py     # Entrenamiento de modelos
        └── predict.py   # Inferencia y evaluación
```

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.12.1+
- pip
- virtualenv (recomendado)

### Pasos

1. **Clonar el repositorio:**
```bash
   git clone <url-del-repo>
   cd Fase2
```

2. **Crear entorno virtual:**
```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
   pip install -r requirements.txt
```

4. **Instalar el paquete en modo desarrollo:**
```bash
   pip install -e .
```

5. **Verificar instalación:**
```bash
   python -c "from fase2 import config; print('✓ Installation successful')"
```

---

## 💻 Uso

### Pipeline Completo

Ejecuta el pipeline completo de ML en orden:

#### 1️⃣ Limpieza de datos
```bash
python -m fase2.dataset
```
**Input:** `data/raw/german_credit_modified.csv`  
**Output:** `data/interim/german_credit_cleaned.csv`

#### 2️⃣ Feature Engineering
```bash
python -m fase2.features
```
**Input:** `data/interim/german_credit_cleaned.csv`  
**Output:** 
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `models/scaler.pkl`

#### 3️⃣ Entrenamiento de Modelos

**Random Forest:**
```bash
python -m fase2.modeling.train --model-name random_forest
```

**Logistic Regression:**
```bash
python -m fase2.modeling.train --model-name logistic_regression
```

**Decision Tree:**
```bash
python -m fase2.modeling.train --model-name decision_tree
```

**Output:** `models/<model_name>.pkl` + metadata JSON

#### 4️⃣ Evaluación
```bash
python -m fase2.modeling.predict --model-path models/random_forest.pkl
```
**Output:** 
- `data/processed/test_predictions.csv`
- `data/processed/test_metrics.json`

#### 5️⃣ Visualizaciones
```bash
python -m fase2.plots
```
**Output:** Gráficos en `reports/figures/`

---

### Uso desde Notebooks

Los notebooks en `notebooks/` demuestran el uso interactivo:
```python
from fase2.dataset import load_raw_data, translate_columns
from fase2.features import prepare_features
from fase2.modeling.train import train_random_forest

# Cargar y preparar datos
df = load_raw_data()
df = translate_columns(df)
X_train, X_test, y_train, y_test, scaler = prepare_features(df)

# Entrenar modelo
model = train_random_forest(X_train, y_train)
```

---

### Uso como Librería Python
```python
from fase2.dataset import load_raw_data, clean_whitespace
from fase2.features import scale_features
from fase2.modeling.predict import load_model, evaluate_model

# Pipeline personalizado
df = load_raw_data("data/raw/german_credit_modified.csv")
df = clean_whitespace(df)
# ... más transformaciones

# Evaluar modelo guardado
model = load_model("models/random_forest.pkl")
metrics = evaluate_model(model, X_test, y_test)
```

---

## 📊 Resultados

### Performance de Modelos

| Modelo                | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-----------------------|----------|-----------|--------|----------|---------|
| Random Forest         | 0.7543   | 0.7421    | 0.8312 | 0.7841   | **0.7821** |
| Logistic Regression   | 0.7412   | 0.7201    | 0.7945 | 0.7556   | 0.7623  |
| Decision Tree         | 0.7298   | 0.7103    | 0.7812 | 0.7441   | 0.7456  |

✅ **Modelo seleccionado:** Random Forest (AUC-ROC = 0.7821 > 0.75)

### Features Más Importantes

1. `checking_account` (0.142)
2. `duration` (0.118)
3. `amount` (0.095)
4. `age` (0.087)
5. `credit_history` (0.076)

*Ver más detalles en `reports/feature_importance.csv`*

---

## 🛠️ Tecnologías

- **Python 3.12.1**
- **Librerías principales:**
  - pandas 2.1.0
  - numpy 1.24.3
  - scikit-learn 1.3.0
  - matplotlib 3.7.2
  - seaborn 0.12.2
- **Herramientas:**
  - Cookiecutter Data Science
  - loguru (logging)
  - typer (CLI)
  - joblib (serialización)

---

## 📈 Siguientes Pasos

- [ ] Implementar ensemble methods (XGBoost, LightGBM)
- [ ] Agregar tracking con MLflow
- [ ] Implementar CI/CD pipeline
- [ ] Crear API REST con FastAPI
- [ ] Dockerizar el proyecto
- [ ] Agregar tests unitarios

---

## 👥 Contribuidores

**Team 34**
- Alejandro Díaz Villagómez

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- Template: [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

---

## 📞 Contacto

Para preguntas sobre el proyecto, contactar a: team34@example.com

---

**Nota:** Este es un proyecto académico con fines educativos.