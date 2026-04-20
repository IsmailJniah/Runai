# RunnAing: Predicción de fatiga acumulada en corredores populares mediante ML

**TFM — Máster en IA, UNIR | Tipo 1 Piloto Experimental**  
Dataset: FitRec (Ni et al., 2019) | Variable objetivo: TRIMP de Banister (1991)

---

## Estado de las fases

| Fase | Descripción | Estado | Notebook | Outputs esperados |
|------|-------------|--------|----------|-------------------|
| 2 | EDA (distribuciones, outliers, correlaciones, span temporal) | ✅ Implementado | `01_eda.ipynb` | `reports/eda/` · `reports/figures/eda_*.png` |
| 3 | Preparación de datos (limpieza + 13 features + TRIMP incremental) | ✅ Implementado | `02_preprocessing.ipynb` | `data/processed/sessions_features.parquet` |
| 4 | Modelado (70/15/15 split + 5 modelos + Optuna 100 trials) | ✅ Implementado | `04_modeling.ipynb` | `models/*.pkl` · `models/optuna_studies/*.json` |
| 5 | Evaluación (MAE/RMSE/R² en test + scatter + residuals) | ✅ Implementado | `05_evaluation.ipynb` | `reports/evaluation/metrics.csv` · `reports/figures/eval_*.png` |
| 6.1 | SHAP (beeswarm · bar · waterfall · force plot) | ✅ Implementado | `06_shap_interpretability.ipynb` | `reports/figures/shap_*.png` |
| 6.2 | ACWR (zonas riesgo, cohorte ≥ 28 días, tabla usuarios en riesgo) | ✅ Implementado | `07_acwr_analysis.ipynb` | `reports/evaluation/acwr_*.csv` · `reports/figures/acwr_*.png` |

> **Pendiente de ejecución**: todos los notebooks requieren el dataset FitRec en `data/raw/`. Ver sección *Dataset* abajo.

---

## Estructura del proyecto

```
/
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA: histogramas, outliers, correlaciones, span temporal
│   ├── 02_preprocessing.ipynb        # Limpieza 7 pasos + 13 features + TRIMP incremental
│   ├── 03_feature_engineering.ipynb  # (legacy) Feature engineering anterior
│   ├── 04_modeling.ipynb             # 70/15/15 split + 5 modelos + Optuna 100 trials
│   ├── 05_evaluation.ipynb           # MAE/RMSE/R² en test + scatter + residuals
│   ├── 06_shap_interpretability.ipynb# SHAP globales (beeswarm, bar) + locales (waterfall, force)
│   └── 07_acwr_analysis.ipynb        # ACWR + zonas + cohorte apta + tabla riesgo
│
├── src/
│   ├── data_loader.py                # Carga JSONL/CSV del dataset FitRec
│   ├── trimp.py                      # TRIMP incremental Banister (b=1.92/1.67 por sexo)
│   ├── features.py                   # 13 features: 9 GPS + 4 cardíacas (hr_mean/max/min, HRV)
│   ├── splits.py                     # group_train_val_test_split (70/15/15) + GroupKFold
│   ├── models.py                     # 5 modelos parametrizables (LR, RF, GB, XGB, LGB)
│   ├── tuning.py                     # Optuna 100 trials, save study pkl + JSON
│   ├── evaluation.py                 # MAE/RMSE/R²/MAPE/Pearson + Wilcoxon-Bonferroni
│   ├── shap_utils.py                 # TreeExplainer + plots
│   └── acwr.py                       # acute7/chronic28 + zonas (sweet spot/precaución/riesgo)
│
├── reports/
│   ├── eda/                          # descriptive_stats.csv, funnel_limpieza.csv, span_por_usuario.csv
│   ├── evaluation/                   # metrics.csv, shap_importance.csv, acwr_zonas.csv
│   └── figures/                      # eda_*.png, eval_*.png, shap_*.png, acwr_*.png
│
├── data/
│   ├── raw/                          # ← COLOCAR fitrec.jsonl aquí (ver instrucciones)
│   └── processed/                    # sessions_features.parquet (generado por notebook 02)
│
├── models/
│   ├── *.pkl                         # Modelos entrenados (generados por notebook 04)
│   ├── best_model.pkl                # Modelo con menor MAE en validación
│   ├── split_meta.json               # Metadatos del split y mejor modelo
│   └── optuna_studies/               # *_study.pkl + *_best_params.json
│
├── tests/
│   ├── test_trimp_formula.py         # Casos conocidos de Banister
│   ├── test_groupkfold_no_leakage.py # Sin leakage entre usuarios
│   ├── test_no_hr_in_features.py     # (legacy) guard no-HR
│   └── test_acwr.py                  # Cohorte y zonas ACWR
│
├── requirements.txt
└── README.md
```

---

## Dataset: FitRec (Ni et al., 2019)

**Cita:**
> Ni, J., Muhlstein, L., & McAuley, J. (2019). Modeling heart rate and activity data
> for personalizing running pace. *WWW '19*, pp. 1343-1353. ACM.

**Descarga:**
1. Accede a: https://sites.google.com/eng.ucsd.edu/fitrec-project/home
2. Descarga `FitRec.tar.gz` (~4 GB comprimido, ~12 GB descomprimido)
3. Descomprime y coloca en:
   ```
   data/raw/fitrec.jsonl   (o fitrec.json / fitrec.csv)
   ```

**Estadísticas del dataset original:**
- ~253.020 sesiones de running · 1.104 usuarios · Periodo 2014–2017

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducción end-to-end

```bash
# 1. Ejecutar tests (no requieren dataset)
pytest tests/ -v

# 2. Ejecutar notebooks en orden (requieren data/raw/fitrec.jsonl)
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_modeling.ipynb      # ~1-4h en CPU
jupyter nbconvert --to notebook --execute notebooks/05_evaluation.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_shap_interpretability.ipynb
jupyter nbconvert --to notebook --execute notebooks/07_acwr_analysis.ipynb
```

---

## Decisiones de diseño clave

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `random_state` | 42 | Reproducibilidad global |
| Split | 70/15/15 por `userId` | Sin leakage entre usuarios |
| Optuna trials | 100 (XGBoost y LightGBM) | Búsqueda bayesiana exhaustiva |
| Optuna sampler | TPE, `seed=42` | Reproducible |
| TRIMP fórmula | Incremental (por timestep) | Más precisa que la agregada por sesión |
| HR en features X | **Incluida** (hr_mean, hr_max, hr_min, hrv_estimate) | Especificación TFM |
| HR_rest | 60 bpm (poblacional) | FitRec no incluye FC_reposo por usuario |
| HR_max | 185 bpm (poblacional) | FitRec no incluye FC_max por usuario |
| ACWR sweet spot | 0.8 ≤ ACWR ≤ 1.3 | Gabbett (2016) |
| ACWR riesgo | ACWR > 1.5 | Gabbett (2016) |
| Cohorte ACWR | span ≥ 28 días | Ventana crónica canónica |

---

## Tests

```bash
pytest tests/ -v
# 35 unit tests: fórmula TRIMP, split sin leakage, features, zonas ACWR
```

---

## Limitaciones documentadas

- FitRec no incluye FC_reposo ni FC_max por usuario → valores poblacionales (60/185 bpm).
- El notebook 04 puede tardar 1–4 horas en CPU (Optuna 100×2 + 3 modelos base).
- Los notebooks 01–07 no son ejecutables sin el dataset FitRec (~12 GB descomprimido).
