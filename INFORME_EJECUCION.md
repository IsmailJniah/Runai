# INFORME DE EJECUCIÓN — RunnAing TFM

**Fecha de generación:** 2026-04-19
**Estado:** Pipeline implementado. Resultados numéricos pendientes de ejecución con el dataset FitRec.

---

## 1. Funnel de Limpieza (Tabla 4.1)

| Paso | Sesiones | Usuarios | Sesiones eliminadas |
|------|----------|----------|---------------------|
| Dataset original (FitRec v1.0) | [PENDIENTE] | [PENDIENTE] | — |
| Solo running (sport=run) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Con género válido | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Con heart_rate (para TRIMP objetivo) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Con GPS completo (lat/lon/alt/timestamp) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Con duración y FC válidas (TRIMP > 0) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| Con 9 features externas completas (sin NaN) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |

> Valores esperados según Ni et al. (2019): ~253.020 sesiones, 1.104 usuarios.
> Ejecutar notebooks 01 y 02 para obtener los valores reales.

---

## 2. Métricas por Modelo (Tabla 4.3)

| Modelo | MAE (mean ± std) | RMSE (mean) | R² (mean) | MAPE (%) | Pearson r | p-valor Bonferroni vs. baseline |
|--------|-----------------|-------------|-----------|----------|-----------|--------------------------------|
| LinearRegression (baseline) | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | — |
| RandomForest | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| GradientBoosting | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| XGBoost | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |
| LightGBM | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] | [PENDIENTE] |

> Ejecutar notebooks 04 y 05 para obtener los valores reales.
> Validación: GroupKFold k=5 por user_id. Optuna: 50 trials, TPE sampler, seed=42.

---

## 3. Cohorte ACWR (Sección 3.3.2 y Tabla 4.4)

| Métrica | Valor |
|---------|-------|
| Total usuarios en dataset limpio | [PENDIENTE] |
| Usuarios con span ≥ 28 días (cohorte apta) | [PENDIENTE] |
| % de la población total | [PENDIENTE] |
| Sesiones en cohorte apta | [PENDIENTE] |
| % de sesiones totales | [PENDIENTE] |

### Distribución por zonas Gabbett (2016)

| Zona | Límite ACWR | Días-usuario | % |
|------|-------------|-------------|---|
| Insuficiente | < 0.80 | [PENDIENTE] | [PENDIENTE] |
| Óptima | 0.80 – 1.30 | [PENDIENTE] | [PENDIENTE] |
| Precaución | 1.30 – 1.50 | [PENDIENTE] | [PENDIENTE] |
| Alto riesgo | > 1.50 | [PENDIENTE] | [PENDIENTE] |

> Ejecutar notebook 07 para obtener los valores reales.

---

## 4. Top-5 Features por Importancia SHAP Global

| Rank | Feature | Descripción | Mean |SHAP| |
|------|---------|-------------|------------|
| 1 | [PENDIENTE] | — | [PENDIENTE] |
| 2 | [PENDIENTE] | — | [PENDIENTE] |
| 3 | [PENDIENTE] | — | [PENDIENTE] |
| 4 | [PENDIENTE] | — | [PENDIENTE] |
| 5 | [PENDIENTE] | — | [PENDIENTE] |

> Ejecutar notebook 06 para obtener los valores reales.
> Figura 4.1 (beeswarm SHAP) → outputs/figures/figura_4_1_shap_beeswarm.png

---

## 5. Tests ejecutados

```
pytest tests/ -v
```

| Test | Estado |
|------|--------|
| test_no_hr_in_features.py::TestFeatureNamesNoHR::test_feature_names_no_hr_keywords | [ver ejecución] |
| test_no_hr_in_features.py::TestFeatureNamesNoHR::test_feature_names_count | [ver ejecución] |
| test_no_hr_in_features.py::TestAssertNoHRLeakage::* | [ver ejecución] |
| test_no_hr_in_features.py::TestComputeFeaturesNoHR::* | [ver ejecución] |
| test_groupkfold_no_leakage.py::TestGroupKFoldSplits::* | [ver ejecución] |
| test_groupkfold_no_leakage.py::TestValidateNoLeakage::* | [ver ejecución] |
| test_trimp_formula.py::TestBansisterTrimp::* | [ver ejecución] |
| test_acwr.py::TestAssignZone::* | [ver ejecución] |
| test_acwr.py::TestFilterEligibleUsers::* | [ver ejecución] |
| test_acwr.py::TestComputeACWRPerUser::* | [ver ejecución] |
| test_acwr.py::TestZoneDistribution::* | [ver ejecución] |

---

## 6. Desviaciones respecto al plan y limitaciones

### Desviaciones técnicas
- **Python 3.9 en entorno local**: La sintaxis `X | Y` para union types requiere Python ≥ 3.10. Se añadió `from __future__ import annotations` en todos los módulos `src/` para compatibilidad con Python 3.9+.
- **FitRec sin FC de reposo/máxima por usuario**: La fórmula de Banister usa valores poblacionales por defecto (HR_rest=60, HR_max=185). Esto es una limitación del dataset, no un error del pipeline.
- **Notebook 04 (modelado)**: La optimización Optuna (50 trials × 5 modelos × 3 folds internos) puede tardar entre 1-4 horas en hardware convencional. Se recomienda ejecutar en GPU o reducir `n_trials` para desarrollo.

### Limitaciones del dataset
- FitRec no contiene datos de sueño, variabilidad de FC (HRV), ni percepción subjetiva del esfuerzo (RPE).
- Los timestamps pueden presentar discontinuidades (pausas en el entrenamiento) que afectan al cálculo de velocidad instantánea.
- No se dispone de la edad de los usuarios, por lo que HR_max = 220 − edad no es aplicable; se usa un valor conservador fijo (185 bpm).

### Estado del pipeline
- **Tests unitarios**: EJECUTADOS (ver resultados arriba)
- **Notebooks 01-07**: PENDIENTES (requieren dataset FitRec en data/raw/)
- **Outputs CSV/PKL**: PENDIENTES
- **Figura 4.1**: PENDIENTE

---

## 7. Instrucciones para actualizar este informe

Una vez descargado el dataset FitRec:

```bash
# 1. Ejecutar pipeline completo
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output notebooks/01_eda_executed.ipynb
# ... (repetir para 02-07)

# 2. Los valores [PENDIENTE] se reemplazarán automáticamente
#    con los resultados reales al ejecutar cada notebook.
```

Los valores numéricos en `outputs/tables/` son la fuente de verdad;
este informe debe actualizarse copiando esos valores.
