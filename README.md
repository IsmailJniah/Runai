# RunnAing: Predicción de fatiga acumulada en corredores populares mediante ML

**TFM — Máster en IA, UNIR | Tipo 1 Piloto Experimental**
Dataset: FitRec (Ni et al., 2019) | Variable objetivo: TRIMP de Banister (1991)

---

## Estructura del proyecto

```
/
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA + span temporal por usuario
│   ├── 02_preprocessing.ipynb        # Limpieza + cálculo de TRIMP objetivo
│   ├── 03_feature_engineering.ipynb  # 9 features externas (sin FC)
│   ├── 04_modeling.ipynb             # Entrenamiento + Optuna + GroupKFold
│   ├── 05_evaluation.ipynb           # Métricas + Wilcoxon-Bonferroni + Tabla 4.3
│   ├── 06_shap_interpretability.ipynb# Figura 4.1 (beeswarm SHAP)
│   └── 07_acwr_analysis.ipynb        # Cohorte apta + zonas Gabbett + Tabla 4.4
├── src/
│   ├── data_loader.py                # Carga JSONL/CSV del dataset FitRec
│   ├── trimp.py                      # Fórmula Banister (b=1.92/1.67 por sexo)
│   ├── features.py                   # 9 features + guard assert anti-FC
│   ├── splits.py                     # GroupKFold por user_id
│   ├── models.py                     # 5 modelos parametrizables
│   ├── tuning.py                     # Optuna 50 trials por modelo
│   ├── evaluation.py                 # MAE/RMSE/R2/MAPE/Pearson + Wilcoxon-Bonferroni
│   ├── shap_utils.py                 # TreeExplainer + plots
│   └── acwr.py                       # acute7/chronic28 + zonas Gabbett
├── outputs/
│   ├── tables/                       # CSVs con Tablas 4.1, 4.3, 4.4
│   ├── figures/                      # Figura 4.1 (beeswarm SHAP)
│   └── models/                       # best_model.pkl
├── tests/
│   ├── test_no_hr_in_features.py     # Assert: FC no entra en X
│   ├── test_groupkfold_no_leakage.py # Assert: sin leakage entre usuarios
│   ├── test_trimp_formula.py         # Casos conocidos de Banister
│   └── test_acwr.py                  # Cohorte y zonas Gabbett
├── requirements.txt
└── README.md
```

---

## Dataset: FitRec (Ni et al., 2019)

**Version exacta usada:** FitRec v1.0 — publicada por el MIT Media Lab en 2019.

**Cita:**
> Ni, J., Muhlstein, L., & McAuley, J. (2019). Modeling heart rate and
> activity data for personalizing running pace. In *Proceedings of The Web
> Conference 2019* (WWW '19), pp. 1343-1353. ACM.

**Descarga:**
1. Accede a: https://sites.google.com/eng.ucsd.edu/fitrec-project/home
2. Descarga `FitRec.tar.gz` (~4 GB comprimido, ~12 GB descomprimido)
3. Descomprime y coloca el archivo principal en:
   ```
   data/raw/fitrec.jsonl   (o fitrec.json)
   ```
4. Crea el directorio si no existe:
   ```bash
   mkdir -p data/raw
   ```

**Formato:** JSONL (una sesion por linea). Campos relevantes:
- `userId`: identificador del usuario
- `sport`: tipo de deporte (`run`, `bike`, etc.)
- `gender`: `male` / `female`
- `heart_rate`: lista de FC por timestep (solo para construir TRIMP, NO para features)
- `timestamp`: lista de epoch-segundos
- `latitude`, `longitude`, `altitude`: series GPS

**Estadisticas del dataset original:**
- ~253.020 sesiones de running
- 1.104 usuarios unicos
- Periodo: 2014-2017

---

## Instalacion

```bash
# Python 3.11 requerido
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproduccion end-to-end

Ejecuta los notebooks en orden (requiere el dataset en `data/raw/`):

```bash
# 1. Activar entorno
source .venv/bin/activate

# 2. Ejecutar tests antes de comenzar
pytest tests/ -v

# 3. Ejecutar notebooks en orden
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_modeling.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_evaluation.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_shap_interpretability.ipynb
jupyter nbconvert --to notebook --execute notebooks/07_acwr_analysis.ipynb

# 4. Outputs generados en:
#    outputs/tables/tabla_4_1_funnel.csv
#    outputs/tables/tabla_4_3_modelos.csv
#    outputs/tables/tabla_4_4_acwr_cohorte.csv
#    outputs/figures/figura_4_1_shap_beeswarm.png
#    outputs/models/best_model.pkl
```

---

## Reglas de reproducibilidad

| Regla | Valor |
|-------|-------|
| `random_state` global | 42 |
| Split | GroupKFold (k=5) por `userId` |
| Optuna trials | 50 por modelo |
| Optuna sampler | TPE con `seed=42` |
| FC en features X | PROHIBIDO -- assert lanza si se detecta |
| Modelos | LinearRegression, RF, GradientBoosting, XGBoost, LightGBM |

---

## Tests

```bash
pytest tests/ -v
```

Los tests verifican:
- Que FC no aparece en la matriz X (`test_no_hr_in_features.py`)
- Que no hay data leakage por usuario (`test_groupkfold_no_leakage.py`)
- Casos conocidos de la formula de Banister (`test_trimp_formula.py`)
- Cohorte y zonas Gabbett (`test_acwr.py`)

---

## Limitaciones conocidas

- Sin el dataset FitRec descargado, los notebooks 01-07 no se pueden ejecutar.
- Los valores en `INFORME_EJECUCION.md` seran [PENDIENTE] hasta ejecutar el pipeline completo.
- FitRec no incluye FC de reposo ni FC maxima por usuario; se usan valores poblacionales (rest=60, max=185).
