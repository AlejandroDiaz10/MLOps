# Matriz de Roles y Responsabilidades – Fase 2 (Proyecto MLOps)

Este documento amplía la documentación generada con MkDocs e integra el criterio de **“Actividades y tareas a realizar por rol”** de la rúbrica.  
Refleja la participación del equipo, la alineación con la metodología MLOps y la trazabilidad de cada etapa del pipeline.

---

## 🎯 Objetivo general

Desarrollar un pipeline de *Machine Learning* reproducible, modular y versionado para el **análisis de riesgo crediticio**, aplicando prácticas de ingeniería de software y control de versiones de datos y modelos.

---

## 👥 Roles del equipo y sus responsabilidades

| Rol | Principales responsabilidades | Código / Artefactos asociados |
|------|-------------------------------|-------------------------------|
| **Data Engineer (DE)** | - Recolecta y limpia los datos.<br>- Estandariza formatos y valida esquemas.<br>- Genera los conjuntos intermedios (`data/interim`). | `fase2/dataset.py`<br>`data/interim/` |
| **Data Scientist (DS)** | - Realiza *feature engineering* y partición *train/test*.<br>- Desarrolla y entrena modelos (`RandomForest`, `LogisticRegression`, `DecisionTree`).<br>- Evalúa métricas (Accuracy, Precision, Recall, F1, AUC).<br>- Genera visualizaciones de desempeño. | `fase2/features.py`<br>`fase2/modeling/train.py`<br>`fase2/modeling/predict.py`<br>`reports/figures/` |
| **Machine Learning Engineer (MLE)** | - Integra todo el pipeline en `run_pipeline.py`.<br>- Implementa herramientas de **tracking (MLflow)** y **versionado (DVC)**.<br>- Asegura reproducibilidad y mantenimiento del código.<br>- Documenta la estructura y flujo del proyecto. | `run_pipeline.py`<br>`dvc.yaml`<br>`Makefile`<br>`pyproject.toml`<br>`README.md` |
| **Equipo (colaborativo)** | - Consolida resultados y métricas comparativas.<br>- Redacta conclusiones y recomendaciones.<br>- Elabora la evidencia final de entrega. | `docs/CONCLUSIONES.md`<br>`docs/EVIDENCIA_CHECKLIST.md`<br>`reports/` |

---

## 🔄 Flujo de trabajo por etapas MLOps

| Etapa | Descripción | Rol responsable |
|--------|--------------|-----------------|
| **1. Prepare** | Limpieza y validación de datos crudos. | Data Engineer |
| **2. Features** | Ingeniería de características y división *train/test*. | Data Scientist |
| **3. Train** | Entrenamiento con *GridSearchCV* y exportación de modelo. | Data Scientist |
| **4. Evaluate** | Evaluación de métricas y generación de reportes visuales. | Data Scientist |
| **5. Tracking y Registry** | Registro de runs, parámetros y métricas en MLflow. | ML Engineer |
| **6. Versionado y Reproducibilidad** | Control de versiones de datos y modelos con DVC. | ML Engineer |
| **7. Documentación y Evidencia** | Reportes, roles y conclusiones finales. | Equipo |

---

## 💼 Beneficios del enfoque por roles

- Clarifica responsabilidades dentro del ciclo de vida del ML.  
- Mejora la trazabilidad y reproducibilidad del pipeline.  
- Aumenta la colaboración multidisciplinaria (DE–DS–MLE).  
- Facilita la evaluación del proyecto bajo la metodología MLOps.

---

## 📦 Archivos y evidencias relacionadas

- `data/interim/`, `data/processed/` → fases DE y DS.  
- `models/` → modelos y metadatos.  
- `reports/figures/` → visualizaciones de resultados.  
- `mlruns/` → experimentos registrados (MLflow).  
- `dvc.yaml` → definición del pipeline reproducible.  
- `README.md` → documentación general y resultados.  

---

**Autoría:**  
Equipo 34 – Proyecto Integrador Fase 2  
*Tecnológico de Monterrey, 2025*
