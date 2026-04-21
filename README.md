# 🏡 California Housing Classification - Machine Learning (Unidad 2)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimizado-green)
![Universidad Mayor](https://img.shields.io/badge/Universidad-Mayor-red)

Este repositorio contiene el proyecto práctico correspondiente a la **Unidad 2: Modelado Supervisado y Evaluación** del curso de Machine Learning de la Escuela de Ingeniería de la **Universidad Mayor**. 

El objetivo principal del proyecto es construir, evaluar y optimizar distintos algoritmos de clasificación binaria para predecir si el valor mediano de una vivienda en California (basado en el dataset de Kaggle) se encuentra por encima o por debajo de la mediana global del mercado.

## 👥 Equipo de Trabajo
* Hector Garrido
* Lu Chu-Ly
* Alvaro Cumsille

---

## 📊 Descripción del Dataset
Se utilizó el clásico conjunto de datos **California Housing**, el cual contiene información sobre censos de bloques de viviendas en California. 
Para este proyecto de clasificación, se creó una variable objetivo binaria (`house_category`) que clasifica las viviendas en dos categorías:
* **Alto (1):** Valor de la vivienda mayor o igual a la mediana.
* **Bajo (0):** Valor de la vivienda menor a la mediana.

---

## ⚙️ Metodología y Pipeline

El proyecto está estructurado en un pipeline secuencial y reproducible:

1. **Preprocesamiento (Heredado de la Unidad 1):**
   * Imputación de valores nulos (mediana) en `total_bedrooms`.
   * Winsorización de variables numéricas a los percentiles [1%, 99%] para tratamiento de outliers.
   * Codificación One-Hot para variables categóricas (`ocean_proximity`).
   * División estratificada del dataset (Train 70% / Validación 15% / Test 15%).
   * Escalado de características numéricas mediante `StandardScaler`.

2. **Modelado Supervisado:**
   Se entrenaron y compararon 6 modelos de aprendizaje automático:
   * Regresión Logística (Penalizaciones L1/Lasso, L2/Ridge y ElasticNet).
   * Random Forest Classifier.
   * XGBoost Classifier.
   * Red Neuronal Perceptrón Multicapa (MLP).

3. **Evaluación de Modelos:**
   Las métricas utilizadas para comparar el rendimiento en el conjunto de validación fueron: **Accuracy, Precision, Recall, F1-Score, AUC-ROC y Estadístico KS**. Se priorizó el **AUC** por su robustez ante la discriminación de clases.

4. **Optimización de Hiperparámetros:**
   El mejor modelo base (XGBoost) fue sometido a una búsqueda exhaustiva utilizando `GridSearchCV` (5-fold CV) para ajustar parámetros clave como `n_estimators`, `max_depth`, `learning_rate` y `subsample`.

---

## 🏆 Resultados Finales

El modelo ganador tras la optimización fue el **XGBoost Optimizado**. Los resultados obtenidos en el conjunto de prueba (**Test Set**, datos nunca antes vistos) demuestran una alta capacidad predictiva y generalización:

| Métrica | Resultado (Test) |
| :--- | :--- |
| **AUC-ROC** | **0.9711** |
| **Accuracy** | 0.9125 |
| **F1-Score** | 0.9128 |
| **Precision** | 0.91 |
| **Recall** | 0.92 |

*Nota: Las curvas ROC y las Matrices de Confusión están disponibles en la ejecución final del notebook.*

---

## 📁 Estructura del Repositorio

```text
repositorio-ML-unidad2/
├── README.md                           # Documentación del proyecto
├── requirements.txt                    # Dependencias y librerías necesarias
├── notebook_unidad2.ipynb              # Código fuente principal (Jupyter Notebook)
├── data/
│   └── housing.csv                     # Dataset original
├── models/
│   └── modelo_final.pkl                # Modelo XGBoost optimizado exportado
└── results/
    └── resultados_comparacion.csv      # Tabla de métricas de todos los modelos
