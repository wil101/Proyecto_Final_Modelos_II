# Predicción de Ingresos con Adult UCI Dataset
## Proyecto Final - Modelos y Simulación II

**Autores:**
- Dayana Ramirez
- Wilmar Osorio
- Santiago Arenas

## 🎬 Video del proyecto  
[Modelos II – Proyecto Final (Exposición)](https://drive.google.com/file/d/1N6FNI8GnlcS0rr_gmx_B-G-k25JGHXKQ/view?usp=drive_link)

Este repositorio contiene la solución completa para el proyecto final del curso *Modelos y Simulación II*. El objetivo principal es desarrollar, comparar y optimizar modelos de Machine Learning para predecir si una persona tiene ingresos superiores a $50K anuales, basándose en características demográficas y laborales del dataset Adult UCI.

## 📋 Descripción del Proyecto

El proyecto aborda un problema de **clasificación binaria desbalanceada** utilizando el dataset estándar Adult UCI (32,561 muestras, 14 características). La solución implementada sigue estrictamente los lineamientos de la guía del curso y cubre las siguientes secciones:

### Sección 0-3: Preparación y Análisis Exploratorio
- **Carga del Dataset**: Descarga automática desde repositorio UCI
- **Limpieza de Datos**: Manejo de valores faltantes (imputación por moda)
- **Codificación**: One-Hot Encoding para variables categóricas (108 características resultantes)
- **Normalización**: StandardScaler aplicado dentro de pipelines CV
- **Manejo de Desbalance**: SMOTE integrado en pipelines para evitar data leakage

### Sección 4: Entrenamiento y Evaluación de Modelos
Implementación completa de **5 familias de modelos de Machine Learning**:

1. **Regresión Logística** (Modelo Paramétrico)
   - Hiperparámetros: C ∈ {0.01, 0.1, 1, 10}, penalty ∈ {l1, l2}, solver=saga
   - Optimización: GridSearchCV con 5-fold StratifiedKFold
   
2. **K-Vecinos más Cercanos** (Modelo No Paramétrico)
   - Hiperparámetros: n_neighbors ∈ {3, 5, 7, 9, 11}, weights ∈ {uniform, distance}
   - Optimización: GridSearchCV con 5-fold StratifiedKFold
   
3. **Random Forest** (Modelo de Ensamble)
   - Hiperparámetros: n_estimators ∈ {100, 200}, max_depth ∈ {10, 20, None}, min_samples_split ∈ {2, 5}
   - Optimización: RandomizedSearchCV (20 iteraciones)
   
4. **Red Neuronal MLP** (Perceptrón Multicapa)
   - Hiperparámetros: hidden_layer_sizes ∈ {(50,), (100,), (50,50)}, activation ∈ {relu, tanh}, alpha ∈ {0.0001, 0.001}
   - Optimización: RandomizedSearchCV (20 iteraciones)
   
5. **SVM con Kernel RBF** (Máquinas de Vectores de Soporte)
   - Hiperparámetros: C ∈ {0.1, 1, 10}, gamma ∈ {0.01, 0.1, 1}
   - **Optimización Especial**: Uso de 40% del dataset con 3-fold CV para reducir tiempo de entrenamiento (2+ horas → 15-20 minutos)

**Métricas de Evaluación:**
- F1-Score (métrica principal para datos desbalanceados)
- ROC-AUC (área bajo la curva ROC)
- Precision y Recall
- Intervalos de Confianza del 95% calculados con desviación estándar de CV
- Visualizaciones: Curvas ROC, Matrices de Confusión, Gráficos comparativos

### Sección 4.1: Tabla de Hiperparámetros
Documentación completa de todos los grids de búsqueda utilizados para cada modelo, incluyendo espacios de búsqueda y estrategias de optimización.

### Sección 5: Reducción de Dimensionalidad

**5.1. Análisis de Importancia de Variables**
- **Mutual Information**: Medida de dependencia entre características y variable objetivo
- **Chi-Cuadrado (χ²)**: Prueba estadística para variables categóricas
- **ANOVA F-value**: Análisis de varianza para características continuas
- Identificación de características de baja relevancia (< percentil 25)
- Visualizaciones: Top 20 mejores, Bottom 20 peores, distribuciones, comparación entre métodos

**5.2. PCA (Reducción Dimensional Lineal)**
- Análisis de varianza explicada acumulada
- Selección automática de componentes (95% de varianza conservada)
- Re-entrenamiento de los 2 mejores modelos con datos transformados
- Comparación de rendimiento: Original vs PCA
- Reducción dimensional lograda: ~50-60% menos dimensiones

**5.3. UMAP (Reducción Dimensional No Lineal)**
- Proyección a espacio de 20 componentes
- Parámetros: n_neighbors=15, min_dist=0.1, metric='euclidean'
- Re-entrenamiento de los 2 mejores modelos con datos transformados
- Comparación de rendimiento: Original vs UMAP
- Capacidad de capturar relaciones no lineales complejas

**5.4. Comparación Global**
- Tabla resumen: Original vs PCA vs UMAP
- Visualizaciones comparativas de F1-Score y ROC-AUC
- Análisis de porcentaje de reducción dimensional
- Recomendaciones automáticas según rendimiento

**Conclusiones de la Sección 5:**
- Identificación de las variables más relevantes para predicción de ingresos
- PCA ofrece interpretabilidad y reducción significativa manteniendo rendimiento
- UMAP captura relaciones no lineales y puede mejorar rendimiento en algunos casos
- Trade-off entre dimensionalidad, interpretabilidad y rendimiento

## 🛠 Tecnologías y Dependencias

El proyecto está desarrollado en **Python 3.8+** y utiliza las siguientes bibliotecas:

| Biblioteca | Versión | Propósito |
|------------|---------|-----------|
| `pandas` | ≥1.3.0 | Manipulación y análisis de datos |
| `numpy` | ≥1.21.0 | Operaciones numéricas y álgebra lineal |
| `scikit-learn` | ≥1.0.0 | Modelos de ML, métricas, preprocesamiento |
| `imbalanced-learn` | ≥0.9.0 | Técnicas de balanceo (SMOTE) |
| `matplotlib` | ≥3.4.0 | Visualizaciones estáticas |
| `seaborn` | ≥0.11.0 | Visualizaciones estadísticas mejoradas |
| `scipy` | ≥1.7.0 | Funciones científicas y estadísticas |
| `joblib` | ≥1.1.0 | Persistencia de modelos |
| `umap-learn` | ≥0.5.0 | Reducción dimensional no lineal |

### Instalación Automática

El notebook incluye una **celda de configuración inicial** que instala todas las dependencias automáticamente:

```python
# Celda 0 del notebook - Instalación automática
import subprocess
import sys

packages = [
    'pandas', 'numpy', 'scikit-learn', 
    'imbalanced-learn', 'matplotlib', 
    'seaborn', 'scipy', 'joblib', 'umap-learn'
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
```

**No es necesario instalar manualmente** las dependencias si ejecutas el notebook completo desde el inicio.

## 🚀 Guía de Uso

### 1. Clonar el Repositorio
```bash
git clone https://github.com/wil101/Proyecto_Final_Modelos_II.git
cd Proyecto_Final_Modelos_II
```

### 2. Ejecutar el Notebook
Abre el archivo `entrenamiento_evaluacion_modelos.ipynb` en:
- **Jupyter Notebook**: `jupyter notebook`
- **Jupyter Lab**: `jupyter lab`
- **VS Code**: Con extensión de Python y Jupyter
- **Google Colab**: Subir el archivo directamente

### 3. Ejecución Secuencial
El notebook está diseñado para ejecutarse **de principio a fin**:

1. **Celda 0**: Instalación automática de dependencias (2-3 minutos)
2. **Celdas 1-10**: Carga y preprocesamiento del dataset (1-2 minutos)
3. **Celdas 11-45**: Entrenamiento de 5 modelos con optimización de hiperparámetros
   - Logistic Regression: ~2-3 minutos
   - k-NN: ~3-4 minutos
   - Random Forest: ~5-7 minutos
   - MLP Neural Network: ~8-10 minutos
   - SVM (optimizado): ~15-20 minutos
4. **Celdas 46-50**: Visualizaciones y comparaciones de rendimiento
5. **Celda 51**: Tabla de hiperparámetros (Sección 4.1)
6. **Celdas 52-60**: Análisis de reducción dimensional (Sección 5)

**Tiempo total estimado**: 40-50 minutos

### 4. Descarga Automática del Dataset
El notebook descarga el dataset directamente desde el repositorio UCI:
```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
```
**No es necesario descargar archivos CSV manualmente**.

### 5. Resultados Generados
Al finalizar la ejecución, obtendrás:
- ✅ Métricas de rendimiento de 5 modelos con intervalos de confianza
- ✅ Visualizaciones: Curvas ROC, matrices de confusión, gráficos comparativos
- ✅ Tabla de hiperparámetros óptimos encontrados
- ✅ Análisis de importancia de variables
- ✅ Comparación de técnicas de reducción dimensional (PCA vs UMAP)
- ✅ Recomendaciones basadas en resultados experimentales

## 📊 Estructura del Notebook

El notebook `entrenamiento_evaluacion_modelos.ipynb` sigue la estructura oficial de la guía del proyecto:

| Sección | Descripción | Celdas Aprox. |
|---------|-------------|---------------|
| **0** | Instalación de dependencias | 1 |
| **1-3** | Carga, exploración y preprocesamiento del dataset | 10 |
| **4** | Entrenamiento y evaluación de 5 modelos de ML | 35 |
| **4.1** | Tabla de hiperparámetros y configuración experimental | 1 |
| **5.1** | Análisis de importancia de variables | 2 |
| **5.2** | PCA: Reducción dimensional lineal | 3 |
| **5.3** | UMAP: Reducción dimensional no lineal | 3 |
| **5.4** | Comparación global de técnicas | 2 |
| **Conclusiones** | Resumen y recomendaciones finales | 1 |

**Total**: ~60 celdas organizadas secuencialmente

### Visualizaciones Incluidas
- 📈 Curvas ROC de los 5 modelos
- 🔢 Matrices de confusión para cada modelo
- 📊 Gráficos de barras comparando métricas (F1, ROC-AUC, Precision, Recall)
- 🎯 Importancia de variables (Top 20, Bottom 20, distribuciones)
- 📉 Varianza explicada acumulada en PCA
- 🔄 Comparación visual Original vs PCA vs UMAP

## 📁 Estructura del Repositorio

```
Proyecto_Final_Modelos_II/
│
├── entrenamiento_evaluacion_modelos.ipynb    # Notebook principal con todo el análisis
├── README.md                                 # Este archivo
├── .gitignore                                # Archivos excluidos de control de versiones
└── archivos/                                 # Directorio de recursos (vacío inicialmente)
```

### Archivos Excluidos (.gitignore)
Para mantener el repositorio ligero y evitar problemas con límites de tamaño de GitHub:
- `modelos_entrenados/` - Modelos serializados (.pkl, .joblib)
- `*.pkl` - Archivos pickle de modelos
- `__pycache__/` - Cache de Python
- `.ipynb_checkpoints/` - Checkpoints de Jupyter

**Nota**: Los modelos entrenados NO están incluidos en el repositorio. Se generan automáticamente al ejecutar el notebook.

## 🔬 Metodología Experimental

### Validación Cruzada Estratificada
- **Estrategia**: StratifiedKFold (k=5 para la mayoría de modelos, k=3 para SVM optimizado)
- **Propósito**: Mantener la proporción de clases en cada fold
- **Beneficio**: Evita sesgo en datasets desbalanceados

### Prevención de Data Leakage
- **Pipeline Integration**: SMOTE y StandardScaler se aplican SOLO en datos de entrenamiento
- **Uso de `ImbPipeline`**: De la biblioteca `imbalanced-learn`
- **Orden del Pipeline**: StandardScaler → SMOTE → Classifier

### Optimización de Hiperparámetros
| Modelo | Método | Iteraciones | Tiempo Aprox. |
|--------|--------|-------------|---------------|
| Logistic Regression | GridSearchCV | 16 combinaciones | 2-3 min |
| k-NN | GridSearchCV | 10 combinaciones | 3-4 min |
| Random Forest | RandomizedSearchCV | 20 muestras | 5-7 min |
| MLP | RandomizedSearchCV | 20 muestras | 8-10 min |
| SVM | RandomizedSearchCV | 6 muestras (40% datos) | 15-20 min |

### Cálculo de Intervalos de Confianza
```
IC 95% = μ ± 1.96 × (σ / √k)
```
Donde:
- μ = media de la métrica en k folds
- σ = desviación estándar
- k = número de folds (5 o 3)

## 🎯 Resultados Esperados

Al ejecutar el notebook completo, se obtienen métricas de rendimiento para:

### Modelos Baseline (Datos Originales)
- 5 modelos entrenados con 108 características
- Métricas con intervalos de confianza del 95%
- Identificación de los 2 mejores modelos

### Reducción Dimensional
- **PCA**: ~50-60% reducción manteniendo 95% de varianza
- **UMAP**: Proyección a 20 componentes capturando relaciones no lineales
- Comparación de rendimiento en ambos espacios reducidos

### Comparación Final
Tabla comparativa mostrando:
- F1-Score Original vs PCA vs UMAP
- ROC-AUC Original vs PCA vs UMAP
- Porcentaje de reducción dimensional
- Recomendación automática del mejor enfoque

## 💡 Aspectos Técnicos Destacados

### 1. Optimización del SVM
El modelo SVM con kernel RBF es computacionalmente costoso. Para hacerlo viable:
- **Estrategia de Muestreo**: Se utiliza 40% del dataset manteniendo estratificación
- **Reducción de CV**: 3-fold en lugar de 5-fold
- **Grid Reducido**: 3×3×1 = 9 combinaciones, 6 iteraciones totales
- **Cache**: `cache_size=1000` MB para acelerar cálculos
- **Resultado**: Reducción de 2+ horas a 15-20 minutos sin sacrificar validez

### 2. Manejo de Desbalance
- **Clase Mayoritaria**: ≤50K (~76%)
- **Clase Minoritaria**: >50K (~24%)
- **Técnica**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Implementación**: Dentro del pipeline de CV para evitar contaminación
- **Beneficio**: Mejora recall y F1-score significativamente

### 3. Reproducibilidad
- **Random Seeds**: Fijados en todas las operaciones aleatorias
- **Instalación Automática**: No requiere configuración manual del entorno

## 📚 Referencias

- **Dataset**: [UCI Machine Learning Repository - Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
- **SMOTE**: Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- **PCA**: Jolliffe, I. T. (2002). "Principal Component Analysis"
- **UMAP**: McInnes, L., et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"
- **Scikit-learn**: Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python"


