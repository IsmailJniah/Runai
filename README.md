# RunAI - Predicción de Esfuerzo en Corredores

TFM Máster en IA - Sistema de predicción de frecuencia cardíaca media en entrenamientos de running utilizando Machine Learning.

## 📊 Información del Modelo

- **Algoritmo**: XGBoost
- **Tarea**: Predecir FC media (bpm) de un entrenamiento
- **Métricas de rendimiento**:
  - R² Score: 0.890
  - MAE: 2.96 bpm
- **Dataset**: FitRec/Endomondo (~20M actividades)
- **Features utilizadas**:
  - speed_mean, speed_max, speed_std
  - altitude_mean, altitude_gain
  - heart_rate_min, heart_rate_max, heart_rate_std
  - duration_min, distance_km
  - gender

## 🏗️ Estructura del Proyecto

```
Runai/
├── api/
│   ├── api.py              # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Container configuration
├── frontend/
│   └── index.html         # Web application (standalone)
├── models/
│   └── modelo_xgboost.pkl # Trained XGBoost model (opcional)
└── README.md
```

## 🚀 Despliegue en Google Cloud Run

### Prerrequisitos

1. Tener instalado Google Cloud SDK
2. Autenticarse en GCP:
```bash
gcloud auth login
```

3. Configurar el proyecto:
```bash
gcloud config set project aesthetic-style-493015-k1
```

### Paso 1: Build y Deploy de la API

Desde el directorio raíz del proyecto (`Runai/`), ejecuta:

```bash
cd api
gcloud builds submit --tag gcr.io/aesthetic-style-493015-k1/runai-api
```

Luego despliega en Cloud Run:

```bash
gcloud run deploy runai-api ^
  --image gcr.io/aesthetic-style-493015-k1/runai-api ^
  --platform managed ^
  --region europe-southwest1 ^
  --port 8080 ^
  --allow-unauthenticated ^
  --memory 1Gi ^
  --cpu 1 ^
  --max-instances 10
```

**Nota para Linux/Mac**: Reemplaza `^` por `\` para continuar líneas.

### Paso 2: Obtener la URL del servicio

Después del deploy, Cloud Run te proporcionará una URL como:
```
https://runai-api-XXXXXXXXXX.europe-southwest1.run.app
```

### Paso 3: Actualizar el Frontend

Edita `frontend/index.html` y actualiza la constante `API_URL` (línea ~450):

```javascript
const API_URL = 'https://runai-api-XXXXXXXXXX.europe-southwest1.run.app/predict';
```

Reemplaza `XXXXXXXXXX` con tu URL real de Cloud Run.

### Paso 4: Probar la API

Verifica que la API esté funcionando:

```bash
curl https://runai-api-XXXXXXXXXX.europe-southwest1.run.app/health
```

Deberías recibir:
```json
{
  "status": "healthy",
  "model": "not_loaded",
  "service": "runai-api"
}
```

## 🧪 Probar la Predicción

### Desde la línea de comandos:

```bash
curl -X POST https://runai-api-XXXXXXXXXX.europe-southwest1.run.app/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"speed_mean\":10.5,\"speed_max\":14.0,\"speed_std\":1.5,\"altitude_mean\":100.0,\"altitude_gain\":50.0,\"heart_rate_min\":110,\"heart_rate_max\":170,\"heart_rate_std\":12.0,\"duration_min\":55.0,\"distance_km\":10.0,\"gender\":\"male\"}"
```

### Desde el navegador:

Abre `frontend/index.html` directamente en tu navegador (doble clic en el archivo).

## 📝 Endpoints de la API

### GET /
Información general de la API

### GET /health
Estado del servicio (health check)

### POST /predict
Realiza una predicción de FC media

**Request body**:
```json
{
  "speed_mean": 10.5,
  "speed_max": 14.0,
  "speed_std": 1.5,
  "altitude_mean": 100.0,
  "altitude_gain": 50.0,
  "heart_rate_min": 110,
  "heart_rate_max": 170,
  "heart_rate_std": 12.0,
  "duration_min": 55.0,
  "distance_km": 10.0,
  "gender": "male"
}
```

**Response**:
```json
{
  "hr_predicha": 158.4,
  "nivel_esfuerzo": "alto",
  "recomendacion": "Entrenamiento intenso. FC predicha: 158.4 bpm...",
  "pace_min_km": 5.71
}
```

## 🎯 Niveles de Esfuerzo

- **Suave** (< 130 bpm): Recuperación y base aeróbica
- **Moderado** (130-144 bpm): Zona aeróbica óptima
- **Alto** (145-159 bpm): Umbral anaeróbico
- **Máximo** (≥ 160 bpm): Alta intensidad

## 🔄 Actualizar el Servicio

Para actualizar la API después de hacer cambios:

```bash
cd api
gcloud builds submit --tag gcr.io/aesthetic-style-493015-k1/runai-api
gcloud run deploy runai-api --image gcr.io/aesthetic-style-493015-k1/runai-api --region europe-southwest1
```

## 📦 Incluir el Modelo Entrenado (Opcional)

Si tienes el archivo `modelo_xgboost.pkl`:

1. Crea la carpeta `models/` en la raíz del proyecto
2. Coloca el archivo `modelo_xgboost.pkl` dentro
3. Modifica el `Dockerfile` para copiar el modelo:

```dockerfile
# Añadir después de COPY api.py .
COPY ../models/modelo_xgboost.pkl /app/models/
```

4. Vuelve a hacer el build y deploy

## 🛠️ Desarrollo Local

### Ejecutar la API localmente:

```bash
cd api
pip install -r requirements.txt
python api.py
```

La API estará disponible en `http://localhost:8080`

### Documentación interactiva:

Una vez ejecutando, visita:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## 📊 Configuración de GCP

- **Project ID**: aesthetic-style-493015-k1
- **Region**: europe-southwest1
- **Service name**: runai-api
- **Puerto**: 8080

## 🐛 Troubleshooting

### Error: "Permission denied"
```bash
gcloud auth login
gcloud config set project aesthetic-style-493015-k1
```

### Error: "API not enabled"
Habilita las APIs necesarias:
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### Ver logs del servicio:
```bash
gcloud run services logs read runai-api --region europe-southwest1
```

## 📄 Licencia

Proyecto académico - TFM Máster en IA

## 👤 Autor

Desarrollado como parte del Trabajo Fin de Máster en Inteligencia Artificial
