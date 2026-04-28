# IntelliDesk — Clasificador de Tickets de Soporte

Sistema de clasificación automática de solicitudes a mesa de ayuda utilizando **Naïve Bayes Multinomial** implementado desde cero, con interfaz web profesional en Flask.

---

## Tabla de Contenidos

1. [Descripción general](#descripción-general)
2. [Arquitectura del sistema](#arquitectura-del-sistema)
3. [Estructura de archivos](#estructura-de-archivos)
4. [Tecnologías utilizadas](#tecnologías-utilizadas)
5. [Instalación](#instalación)
6. [Uso](#uso)
7. [Cambiar dataset o categorías](#cambiar-dataset-o-categorías)
8. [API REST](#api-rest)
9. [Algoritmo implementado](#algoritmo-implementado)

---

## Descripción general

IntelliDesk clasifica automáticamente tickets de soporte técnico en categorías predefinidas usando aprendizaje automático supervisado. El modelo se entrena con un dataset CSV y sirve predicciones en tiempo real a través de una interfaz web SPA (Single Page Application).

**Capacidades:**
- Clasificación instantánea de texto con probabilidades por categoría
- Validación del modelo con K-Folds Cross Validation (K=5) — **99.76% accuracy**
- Vista de estadísticas del modelo (accuracy, F1, distribución por clase)
- Historial de tickets en la sesión con filtros por estado
- Dashboard de resumen con conteo por categoría

---

## Arquitectura del sistema

```
Usuario (navegador)
       │
       │  HTTP / JSON
       ▼
┌─────────────────────┐
│   Flask (app.py)    │   ← Servidor web y API REST
│   Puerto 5000       │
└────────┬────────────┘
         │ carga al inicio
         ▼
┌─────────────────────┐
│  naive_bayes.pkl    │   ← Modelo entrenado (pickle)
│  (model/)           │
└────────┬────────────┘
         │ usa
         ▼
┌─────────────────────────────────────────────────────┐
│                    src/                             │
│  preprocessor.py → bag_of_words.py → naive_bayes.py│
│  evaluator.py   ← train.py → config.py             │
└─────────────────────────────────────────────────────┘
```

**Flujo de clasificación:**
```
Texto del usuario
      │
      ▼ preprocessor.preprocess()
  Tokens limpios
      │
      ▼ classifier.predict_proba()
  Probabilidades log
      │
      ▼ softmax → porcentajes
  Categoría predicha + probabilidades
      │
      ▼ JSON → frontend (script.js)
  UI actualizada con resultado
```

---

## Estructura de archivos

```
proyecto-mesa-ayuda/
│
├── README.md                        # Este archivo
├── requirements.txt                 # Dependencias Python (Flask, NLTK)
├── setup_nltk.py                    # Descarga recursos NLTK al iniciar
│
├── src/                             # Lógica de IA (implementada desde cero)
│   ├── config.py                    # *** CONFIGURACIÓN CENTRAL ***
│   │                                #     Dataset, columnas y categorías
│   ├── preprocessor.py              # Pipeline: limpieza → tokens → stopwords → stemming
│   ├── bag_of_words.py              # Construcción del vocabulario (Bag of Words)
│   ├── naive_bayes.py               # Clasificador Naïve Bayes Multinomial
│   ├── evaluator.py                 # K-Folds Cross Validation y métricas
│   ├── train.py                     # Script de entrenamiento + evaluación
│   ├── predict.py                   # Prueba rápida de predicción desde consola
│   ├── explore_data.py              # Exploración del dataset (EDA)
│   └── filter_dataset.py            # Utilidad para filtrar/balancear dataset
│
├── web/                             # Interfaz web (Flask SPA)
│   ├── app.py                       # Servidor Flask + endpoints API
│   ├── templates/
│   │   └── index.html               # SPA: Dashboard, Nuevo Ticket, Historial, Modelo
│   └── static/
│       ├── config.js                # *** CONFIGURACIÓN FRONTEND ***
│       │                            #     Mismo array de categorías (sincronizado con config.py)
│       ├── script.js                # Lógica de la SPA (navegación, clasificación, historial)
│       └── style.css                # Diseño oscuro con acentos fosforescentes
│
├── data/
│   ├── bitext_customer_support.csv  # Dataset de entrenamiento (CSV)
│   └── Documento_Proyecto_IntelliDesk.docx  # Documentación del proyecto
│
└── model/
    └── naive_bayes_model.pkl        # Modelo entrenado (generado por train.py)
```

---

## Tecnologías utilizadas

| Capa | Tecnología | Versión | Propósito |
|------|-----------|---------|-----------|
| **Backend** | Python | 3.11+ | Lenguaje principal |
| **Web framework** | Flask | 3.0.0 | Servidor HTTP y API REST |
| **NLP** | NLTK | 3.8.1 | Stopwords y stemming (PorterStemmer) |
| **ML** | Implementación propia | — | Naïve Bayes, K-Folds, BoW |
| **Frontend** | HTML5 + CSS3 + JS | — | SPA sin frameworks |
| **Serialización** | pickle | stdlib | Persistencia del modelo |

**Sin dependencias externas de ML**: scikit-learn, pandas, numpy — todo implementado manualmente.

---

## Instalación

### Requisitos previos
- Python 3.11 o superior
- pip

### Pasos

```bash
# 1. Clonar o descargar el proyecto
git clone https://github.com/danielxiquin/proyectoIA

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar el entorno virtual
#    macOS / Linux:
source venv/bin/activate
#    Windows:
venv\Scripts\activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Descargar recursos NLTK (solo la primera vez)
python setup_nltk.py
```

### Entrenar el modelo

```bash
# Desde la raíz del proyecto:
python src/train.py
```

Esto genera `model/naive_bayes_model.pkl` con el clasificador entrenado y las métricas de evaluación K-Folds.

---

## Uso

### Iniciar el servidor web

```bash
# Desde la raíz del proyecto (con venv activo):
python web/app.py
```

Abrir en el navegador: **http://localhost:5000**

### Vistas disponibles

| Vista | Descripción |
|-------|-------------|
| **Dashboard** | Resumen de tickets por estado y categoría |
| **New Ticket** | Formulario de clasificación + panel de resultados IA |
| **AI Model** | Estadísticas del modelo: accuracy, F1, métricas por clase |
| **History** | Tabla de tickets guardados con filtros |

### Probar clasificación desde consola

```bash
python src/predict.py
```

---

## Cambiar dataset o categorías

El proyecto centraliza toda la configuración en **dos archivos**. Si el dataset cambia, edita solo estos:

### Backend: `src/config.py`

```python
# 1. Cambiar archivo de datos
DATASET_FILE = "nuevo_dataset.csv"   # Debe estar en data/

# 2. Cambiar nombres de columnas del CSV
COL_TEXT  = "texto"     # Columna con el texto del ticket
COL_LABEL = "categoria" # Columna con la etiqueta

# 3. Cambiar categorías
CATEGORIES = [
    {"key": "TECNICO",   "display": "Soporte Técnico", "color": "#FFE600"},
    {"key": "FACTURA",   "display": "Facturación",     "color": "#39FF14"},
    # Agrega o elimina categorías aquí
    # key = valor exacto en el CSV
    # display = nombre a mostrar en la UI
    # color = color fosforescente en hex
]
```

### Frontend: `web/static/config.js`

```javascript
// Debe reflejar el mismo array que src/config.py
const CATEGORIES = [
  { key: "TECNICO",  display: "Soporte Técnico", color: "#FFE600" },
  { key: "FACTURA",  display: "Facturación",     color: "#39FF14" },
  // ...
];
```

### Aplicar cambios

```bash
# Reentrenar con el nuevo dataset
python src/train.py

# Reiniciar el servidor
python web/app.py
```

---

## API REST

### `POST /classify`

Clasifica un texto en una categoría.

**Request:**
```json
{
  "text": "My laptop won't turn on after the update",
  "subject": "Technical problem"
}
```

**Response:**
```json
{
  "predicted_class": "TECHNICAL",
  "display_name": "Technical Support",
  "color": "#FFE600",
  "probabilities": {
    "Technical Support": 87.3,
    "Billing": 5.2,
    "Account": 3.1,
    "..."
  },
  "timestamp": "2026-04-21 10:30:00",
  "tokens_used": 9
}
```

### `GET /model-stats`

Retorna métricas de evaluación del modelo entrenado.

**Response:**
```json
{
  "has_eval": true,
  "accuracy": 0.9976,
  "macro_f1": 0.9976,
  "k_folds": 5,
  "n_docs": 9750,
  "vocab_size": 4821,
  "classes": ["ACCOUNT", "BILLING", ...],
  "per_class": {
    "BILLING": { "precision": 0.998, "recall": 0.997, "f1": 0.998 }
  },
  "docs_per_class": { "BILLING": 750 }
}
```

### `GET /health`

Estado del servidor y modelo cargado.

---

## Algoritmo implementado

### Pipeline de preprocesamiento (`preprocessor.py`)

```
Texto crudo
  → lowercase + eliminar caracteres especiales
  → tokenización por espacios
  → eliminación de stopwords (NLTK english)
  → stemming (PorterStemmer)
  → lista de tokens
```

### Naïve Bayes Multinomial (`naive_bayes.py`)

Probabilidad de clase dado un documento:

```
P(clase | doc) ∝ P(clase) × ∏ P(token | clase)
```

Con Laplace Smoothing (α=1.0) para tokens no vistos:

```
P(token | clase) = (frecuencia(token, clase) + α) / (total_tokens_clase + α × |vocabulario|)
```

Implementado en **log-probabilidades** para evitar underflow numérico.

### Evaluación K-Folds (`evaluator.py`)

- K=5 folds estratificados
- Métricas por fold: Precision, Recall, F1-score por clase
- Resultado final: promedio de los 5 folds
- Matriz de confusión global acumulada

---

---

## Resultados K-Folds Reales

El modelo fue validado con K-Folds Cross Validation (K=5) en el dataset completo de 9,750 tickets:

```
Fold 1: Accuracy=0.9979 | Macro F1=0.9980
Fold 2: Accuracy=0.9969 | Macro F1=0.9968
Fold 3: Accuracy=0.9964 | Macro F1=0.9963
Fold 4: Accuracy=0.9974 | Macro F1=0.9975
Fold 5: Accuracy=0.9995 | Macro F1=0.9995

────────────────────────────────────────
Accuracy promedio : 0.9976 ± 0.0011 (99.76%)
Macro F1 promedio : 0.9976
```

Estos números reflejan un modelo **robusto y de alta precisión** que generaliza bien a nuevos datos.

---

*Universidad Rafael Landívar · Inteligencia Artificial · 2026*
