# RunnAing: Predicción de fatiga acumulada en corredores populares mediante ML

**TFM — Máster en IA, UNIR | Tipo 1 Piloto Experimental**  
Dataset: FitRec (Ni et al., 2019) | Variable objetivo: TRIMP de Banister (1991)

---

## Estado de las fases

| Fase | Descripción | Estado | Notebook | Outputs esperados |
|------|-------------|--------|----------|-------------------|
| 2 | EDA (distribuciones, outliers, correlaciones, span temporal) | ✅ Implementado | `01_eda.ipynb` | `reports/eda/` · `reports/figures/eda_*.png` |
| 3 | Preparación de datos (9 features externas + TRIMP incremental con 0.64) | ✅ Implementado | `02_preprocessing.ipynb` | `data/processed/sessions_features.parquet` |
| 4 | Modelado (70/15/15 split + 5 modelos + Optuna 100 trials × 5) | ✅ Implementado | `04_modeling.ipynb` | `models/*.pkl` · `models/optuna_studies/*.json` |
| 5 | Evaluación (MAE/RMSE/R² en test + scatter + residuals + subgrupos) | ✅ Implementado | `05_evaluation.ipynb` | `reports/evaluation/metrics.csv` · `reports/figures/eval_*.png` |
| 6.1 | SHAP (beeswarm · bar · waterfall · force plot) | ✅ Implementado | `06_shap_interpretability.ipynb` | `reports/figures/shap_*.png` |
| 6.2 | ACWR (media 7d/28d, cohorte ≥ 28 días, zonas Gabbett) | ✅ Implementado | `07_acwr_analysis.ipynb` | `reports/evaluation/acwr_*.csv` · `reports/figures/acwr_*.png` |
| 7 | Dashboard Streamlit (CSV → TRIMP → ACWR → SHAP individual) | ✅ Implementado | `app.py` | UI interactiva |

> **Pendiente de ejecución**: todos los notebooks requieren el dataset FitRec en `data/raw/`. Ver sección *Dataset* abajo.

---

## Estructura del proyecto

```
/
├── app.py                            # ← Dashboard Streamlit (arrancar con: streamlit run app.py)
│
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA: histogramas, outliers, correlaciones, span temporal
│   ├── 02_preprocessing.ipynb        # Limpieza 7 pasos + 9 features externas + TRIMP (sin FC en X)
│   ├── 04_modeling.ipynb             # 70/15/15 split + 5 modelos + Optuna 100 trials × 5
│   ├── 05_evaluation.ipynb           # MAE/RMSE/R² en test + scatter + residuals + subgrupos
│   ├── 06_shap_interpretability.ipynb# SHAP globales (beeswarm, bar) + locales (waterfall, force)
│   └── 07_acwr_analysis.ipynb        # ACWR (media 7d/28d) + zonas + cohorte + tabla riesgo
│
├── src/
│   ├── data_loader.py                # Carga JSONL/CSV del dataset FitRec
│   ├── trimp.py                      # TRIMP incremental Banister con factor 0.64 (b=1.92/1.67)
│   ├── features.py                   # 9 features externas (GPS/mecánicas) — sin FC (OE3)
│   ├── splits.py                     # group_train_val_test_split (70/15/15) + GroupKFold
│   ├── models.py                     # 5 modelos parametrizables (LR, RF, GB, XGB, LGB)
│   ├── tuning.py                     # Optuna 100 trials × 5 modelos, early stopping XGB/LGB
│   ├── evaluation.py                 # MAE/RMSE/R²/MAPE/Pearson + Wilcoxon-Bonferroni
│   ├── shap_utils.py                 # TreeExplainer + plots (beeswarm, bar, waterfall)
│   └── acwr.py                       # media(7d)/media(28d) + zonas Gabbett (2016)
│
├── legacy/
│   └── Notebook-modelo.ipynb         # Pipeline exploratorio anterior (distinto objetivo, no usar)
│
├── reports/
│   ├── eda/                          # descriptive_stats.csv, funnel_limpieza.csv
│   ├── evaluation/                   # metrics.csv, shap_importance.csv, acwr_zonas.csv
│   └── figures/                      # eda_*.png, eval_*.png, shap_*.png, acwr_*.png
│
├── data/
│   ├── raw/                          # ← COLOCAR fitrec.jsonl aquí (ver instrucciones)
│   └── processed/                    # sessions_features.parquet (generado por notebook 02)
│
├── models/
│   ├── *.pkl                         # Modelos entrenados (generados por notebook 04)
│   ├── best_model.pkl                # Modelo con menor MAE en validación (usado por app.py)
│   ├── split_meta.json               # Metadatos del split y mejor modelo
│   └── optuna_studies/               # *_study.pkl + *_best_params.json (× 5 modelos)
│
├── tests/
│   ├── test_trimp_formula.py         # Banister (0.64, b diferenciado, versión incremental)
│   ├── test_groupkfold_no_leakage.py # Sin leakage entre usuarios
│   ├── test_no_hr_in_features.py     # Guard: FC excluida de X (ruta build_feature_matrix)
│   └── test_acwr.py                  # ACWR≈1.0 en carga constante, cohorte y zonas
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
| `random_state` | 42 | Reproducibilidad global (splits, modelos, Optuna) |
| Split | 70/15/15 por `userId` | Sin leakage entre usuarios |
| Features X | **9 externas** (GPS/mecánicas, sin FC) | OE3 del TFM — predictor sin FC |
| Optuna trials | 100 × 5 modelos | Búsqueda bayesiana exhaustiva; early stopping en XGB/LGB |
| Optuna sampler | TPE, `seed=42` | Reproducible |
| TRIMP fórmula | Incremental con factor **0.64** | Banister (1991) canónico |
| HR_rest | 60 bpm (poblacional) | FitRec no incluye FC_reposo por usuario |
| HR_max | 185 bpm (poblacional) | FitRec no incluye FC_max por usuario |
| ACWR | media(7d) / media(28d) | Gabbett (2016) canónico; ACWR≈1.0 en carga constante |
| ACWR zona óptima | 0.8 ≤ ACWR ≤ 1.3 | Gabbett (2016) |
| ACWR zona riesgo | ACWR > 1.5 | Gabbett (2016) |
| Cohorte ACWR | span ≥ 28 días | Ventana crónica canónica |

---

## Dashboard Streamlit

```bash
# Requiere haber ejecutado el pipeline hasta el notebook 05 (genera best_model.pkl)
streamlit run app.py
```

Flujo del dashboard:
1. **Carga CSV** con sesiones (duración, distancia, velocidad, desnivel, ritmo, sexo, fecha, userId).
2. **Predicción TRIMP** por sesión — histograma + tabla.
3. **ACWR diario** por usuario (media 7d / media 28d, zonas Gabbett).
4. **Explicación SHAP** individual (waterfall plot) para cualquier sesión.

---

## Tests

```bash
pytest tests/ -v
# 48 unit tests: TRIMP (agregada + incremental), leakage HR, split, ACWR
```

---

## Reproducción end-to-end (pasos exactos)

```bash
# 1. Clonar y preparar entorno
git clone <url>
cd RunnAing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Descargar dataset FitRec (~12 GB descomprimido)
#    https://sites.google.com/eng.ucsd.edu/fitrec-project/home
#    Colocar en data/raw/fitrec.jsonl

# 3. Ejecutar pipeline en orden
pytest tests/ -v                                                         # 48 tests, sin dataset
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_modeling.ipynb    # ~2-5h en CPU
jupyter nbconvert --to notebook --execute notebooks/05_evaluation.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_shap_interpretability.ipynb
jupyter nbconvert --to notebook --execute notebooks/07_acwr_analysis.ipynb

# 4. Dashboard
streamlit run app.py
```

---

## Limitaciones documentadas

- FitRec no incluye FC_reposo ni FC_max por usuario → valores poblacionales (60/185 bpm).
- El notebook 04 puede tardar 2–5 horas en CPU (Optuna 100 trials × 5 modelos).
- Los notebooks 01–07 no son ejecutables sin el dataset FitRec (~12 GB descomprimido).
- `legacy/Notebook-modelo.ipynb`: pipeline exploratotio anterior con target distinto (HR, no TRIMP). No forma parte del TFM.
