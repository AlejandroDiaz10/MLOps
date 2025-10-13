# 🏦 South German Credit Risk Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-blue.svg)](https://dvc.org/)

Proyecto académico de Machine Learning para clasificación de riesgo crediticio utilizando el dataset **South German Credit** (UCI ML Repository). Implementación completa de pipeline MLOps desde la adquisición de datos hasta el entrenamiento y evaluación de modelos.

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema de clasificación binaria para evaluar el riesgo crediticio de solicitantes de préstamos bancarios. Utilizando el dataset histórico de South German Credit con 1,000 solicitudes, desarrollamos un pipeline completo de Machine Learning siguiendo mejores prácticas de MLOps.

### Objetivo

Predecir si un solicitante de crédito es:
- **Clase 1**: Buen pagador (good credit)
- **Clase 0**: Mal pagador (bad credit)

### Características Principales

- ✅ **Pipeline completo de datos**: Desde raw data hasta modelos entrenados
- ✅ **Versionado con DVC**: Trazabilidad completa de datos y modelos
- ✅ **3 modelos ML**: Regresión Logística, Árbol de Decisión, Random Forest
- ✅ **Validación rigurosa**: 5-fold cross-validation + 20% holdout test
- ✅ **Análisis de interpretabilidad**: Feature importance y coeficientes
- ✅ **Visualizaciones profesionales**: Gráficas de alta calidad para análisis

---

## 🔧 Requisitos

### Software

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git
- DVC (opcional, para versionado de datos)

### Bibliotecas Principales

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

Ver archivo `requirements.txt` para la lista completa de dependencias.

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/AlejandroDiaz10/MLOps
cd south-german-credit
```

### 2. Crear entorno virtual (recomendado)

```bash
# Con venv
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# O con conda
conda create -n credit-risk python=3.8
conda activate credit-risk
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 👥 Equipo

Este proyecto fue desarrollado por un equipo multidisciplinario siguiendo la metodología MLOps:

|  Nombre completo                |     Matrícula     | Rol               | Responsabilidades                                        |
| :--------------------------:    |:-----------------:|:-----------------:|:--------------------------------------------------------:|
| Emilio Contreras Téllez         |  A01111353        | ML Engineer       | Selección de modelos, entrenamiento, optimización        |
| Claudio Luis Del Valle Azuara   |  A01795773        | DevOps            | Git, control de versiones, gestión de artifacts          |  
| Alejandro Díaz Villagómez       |  A01276769        | Data Scientist    | EDA, análisis estadístico, interpretación de resultados  |
| Guillermo Herrera Acosta        |  A01400835        | Software Engineer | Código modular, testing, documentación técnica           |
| Ivan Troy Santaella Martinez    |  A01120515        | Data Engineer     | Pipeline de datos, versionado DVC, validación de calidad |

---

## 📚 Referencias

### Dataset

- **Fuente**: UCI Machine Learning Repository
- **Título**: South German Credit (Update)
- **Autor**: Grömping, U. (2020)
- **DOI**: [10.24432/C5QG8F](https://doi.org/10.24432/C5QG8F)
- **URL**: [https://archive.ics.uci.edu/dataset/573/south+german+credit+update](https://archive.ics.uci.edu/dataset/573/south+german+credit+update)

### Documentación Técnica

- Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set*. Beuth University of Applied Sciences Berlin. [PDF](http://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf)

### Herramientas

- [scikit-learn Documentation](https://scikit-learn.org/)
- [DVC Documentation](https://dvc.org/doc)
- [Machine Learning Canvas](https://www.ownml.co/machine-learning-canvas)

---

## 🤝 Contribuciones

Este es un proyecto académico. Para sugerencias o mejoras:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📝 Notas Adicionales

### Reproducibilidad

El proyecto utiliza `random_state=42` en todas las operaciones aleatorias para garantizar reproducibilidad completa.

### Versionado de Datos

Los datos están versionados con DVC en dos puntos:
- **v1**: Después de limpieza (`german_credit_cleaned_v1.csv`)
- **v2**: Después de preprocessing (`german_credit_processed_v2.csv`)

### Comandos DVC Útiles

```bash
# Descargar datos versionados
dvc pull

# Ver cambios en datos
dvc status

# Actualizar datos
dvc add data/archivo.csv
git add data/archivo.csv.dvc
git commit -m "Update data"
dvc push
```

---