from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(
    title="RunAI - Predicción de Esfuerzo en Corredores",
    description="API para predecir frecuencia cardíaca media usando XGBoost",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class PredictionInput(BaseModel):
    speed_mean: float = Field(..., description="Velocidad media (km/h)", ge=0)
    speed_max: float = Field(..., description="Velocidad máxima (km/h)", ge=0)
    speed_std: float = Field(..., description="Desviación estándar velocidad", ge=0)
    altitude_mean: float = Field(..., description="Altitud media (m)")
    altitude_gain: float = Field(..., description="Ganancia de altitud (m)", ge=0)
    heart_rate_min: int = Field(..., description="FC mínima (bpm)", ge=40, le=220)
    heart_rate_max: int = Field(..., description="FC máxima (bpm)", ge=40, le=220)
    heart_rate_std: float = Field(..., description="Desviación estándar FC", ge=0)
    duration_min: float = Field(..., description="Duración (minutos)", ge=0)
    distance_km: float = Field(..., description="Distancia (km)", ge=0)
    gender: str = Field(..., description="Género (male/female)")

    class Config:
        json_schema_extra = {
            "example": {
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
        }

# Modelo de salida
class PredictionOutput(BaseModel):
    hr_predicha: float
    nivel_esfuerzo: str
    recomendacion: str
    pace_min_km: float

# Cargar modelo
MODEL_PATH = Path(__file__).parent.parent / "models" / "modelo_xgboost.pkl"
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Modelo cargado desde {MODEL_PATH}")
    except FileNotFoundError:
        print(f"⚠ Advertencia: Modelo no encontrado en {MODEL_PATH}")
        print("  La API funcionará en modo demo con predicciones simuladas")
    except Exception as e:
        print(f"⚠ Error al cargar modelo: {e}")

def get_nivel_esfuerzo(hr: float) -> str:
    """Determina el nivel de esfuerzo según la FC predicha"""
    if hr < 130:
        return "suave"
    elif hr < 145:
        return "moderado"
    elif hr < 160:
        return "alto"
    else:
        return "maximo"

def get_recomendacion(nivel: str, hr: float) -> str:
    """Genera recomendación según el nivel de esfuerzo"""
    recomendaciones = {
        "suave": f"Ritmo de recuperación ideal. FC predicha: {hr:.1f} bpm. Perfecto para entrenamientos de base aeróbica y recuperación activa.",
        "moderado": f"Zona aeróbica óptima. FC predicha: {hr:.1f} bpm. Ideal para mejorar resistencia y quemar grasas de forma eficiente.",
        "alto": f"Entrenamiento intenso. FC predicha: {hr:.1f} bpm. Zona de umbral anaeróbico, mejora tu velocidad pero requiere buena recuperación.",
        "maximo": f"Esfuerzo máximo. FC predicha: {hr:.1f} bpm. Zona de alta intensidad, limita este tipo de entrenamientos y asegura descanso adecuado."
    }
    return recomendaciones.get(nivel, "Nivel de esfuerzo no determinado")

def calculate_pace(speed_kmh: float) -> float:
    """Calcula el pace (min/km) desde velocidad (km/h)"""
    if speed_kmh <= 0:
        return 0.0
    return 60.0 / speed_kmh

@app.get("/")
async def root():
    return {
        "message": "RunAI API - Predicción de Esfuerzo en Corredores",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Realizar predicción",
            "/health": "GET - Estado del servicio",
            "/docs": "GET - Documentación interactiva"
        }
    }

@app.get("/health")
async def health():
    """Endpoint de salud para Cloud Run"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model": model_status,
        "service": "runai-api"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    """
    Predice la frecuencia cardíaca media de un entrenamiento
    
    Modelo: XGBoost
    Métricas: R²=0.890, MAE=2.96 bpm
    Dataset: FitRec/Endomondo
    """
    try:
        # Validar género
        if data.gender.lower() not in ["male", "female"]:
            raise HTTPException(
                status_code=400,
                detail="El género debe ser 'male' o 'female'"
            )
        
        # Validar coherencia de datos
        if data.heart_rate_min >= data.heart_rate_max:
            raise HTTPException(
                status_code=400,
                detail="heart_rate_min debe ser menor que heart_rate_max"
            )
        
        if data.speed_mean > data.speed_max:
            raise HTTPException(
                status_code=400,
                detail="speed_mean no puede ser mayor que speed_max"
            )
        
        # Preparar datos para predicción
        # Convertir género a numérico (male=1, female=0)
        gender_numeric = 1 if data.gender.lower() == "male" else 0
        
        # Crear DataFrame con las features en el orden correcto
        features = pd.DataFrame({
            'speed_mean': [data.speed_mean],
            'speed_max': [data.speed_max],
            'speed_std': [data.speed_std],
            'altitude_mean': [data.altitude_mean],
            'altitude_gain': [data.altitude_gain],
            'heart_rate_min': [data.heart_rate_min],
            'heart_rate_max': [data.heart_rate_max],
            'heart_rate_std': [data.heart_rate_std],
            'duration_min': [data.duration_min],
            'distance_km': [data.distance_km],
            'gender': [gender_numeric]
        })
        
        # Realizar predicción
        if model is not None:
            hr_predicha = float(model.predict(features)[0])
        else:
            # Modo demo: estimación basada en reglas simples
            # Aproximación: FC media ≈ (FC_min + FC_max) / 2 + ajuste por intensidad
            intensidad_factor = (data.speed_mean / 12.0) * 10  # Ajuste por velocidad
            hr_predicha = (data.heart_rate_min + data.heart_rate_max) / 2 + intensidad_factor
            hr_predicha = min(max(hr_predicha, 60), 200)  # Limitar a rango razonable
        
        # Redondear a 1 decimal
        hr_predicha = round(hr_predicha, 1)
        
        # Determinar nivel de esfuerzo
        nivel = get_nivel_esfuerzo(hr_predicha)
        
        # Generar recomendación
        recomendacion = get_recomendacion(nivel, hr_predicha)
        
        # Calcular pace
        pace = round(calculate_pace(data.speed_mean), 2)
        
        return PredictionOutput(
            hr_predicha=hr_predicha,
            nivel_esfuerzo=nivel,
            recomendacion=recomendacion,
            pace_min_km=pace
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
