"""
fix_and_run.py — Ejecutar desde Jupyter con:  %run fix_and_run.py
O pegando el contenido en una celda.

Pasos:
  1. Parchea src/data_loader.py para soportar el formato de endomondoHR.json
  2. Instala pyarrow si no está disponible
  3. Verifica que el dataset carga correctamente
"""

import subprocess
import sys
from pathlib import Path

# ── 0. Ruta base del proyecto ────────────────────────────────────────────────
BASE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
while not (BASE / "src").exists() and BASE != BASE.parent:
    BASE = BASE.parent
print(f"Raíz del proyecto: {BASE}")

# ── 1. Sobrescribir data_loader.py con la versión corregida ──────────────────
DATA_LOADER = BASE / "src" / "data_loader.py"

FIXED_CONTENT = '''"""
data_loader.py — Carga del dataset Endomondo / FitRec.

Soporta dos formatos por línea:
  - JSON estándar  {"key": value}   (FitRec original)
  - Python dict    {\'key\': value}  (endomondoHR.json)
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LIST_FIELDS = ["heart_rate", "longitude", "latitude", "altitude", "timestamp"]
TARGET_SPORT = "run"


def load_fitrec_jsonl(path: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    records = []
    bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    records.append(ast.literal_eval(line))
                except Exception as exc:
                    bad += 1
                    logger.debug("Línea %d malformada, se omite: %s", i, exc)

    if bad:
        logger.warning("%d líneas malformadas omitidas de %d totales", bad, i + 1)

    df = pd.DataFrame(records)
    logger.info("Cargadas %d sesiones desde %s", len(df), path)
    return df


def load_fitrec_csv(path: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    df = pd.read_csv(path, nrows=max_rows)
    for col in _LIST_FIELDS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(_safe_parse_list)

    logger.info("Cargadas %d sesiones desde CSV %s", len(df), path)
    return df


def _safe_parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def filter_running_sessions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "sport" in df.columns:
        df = df[df["sport"].str.lower() == TARGET_SPORT].copy()
    if "userId" in df.columns:
        df = df[df["userId"].notna()].copy()
    logger.info("Sesiones de running con userId válido: %d", len(df))
    return df


def auto_detect_and_load(data_dir: str | Path = "data/raw",
                         max_rows: Optional[int] = None) -> pd.DataFrame:
    data_dir = Path(data_dir)
    candidates = list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.json"))
    if candidates:
        return load_fitrec_jsonl(candidates[0], max_rows=max_rows)

    candidates_csv = list(data_dir.glob("*.csv"))
    if candidates_csv:
        return load_fitrec_csv(candidates_csv[0], max_rows=max_rows)

    raise FileNotFoundError(
        f"No se encontró ningún archivo de datos en {data_dir}."
    )
'''

DATA_LOADER.write_text(FIXED_CONTENT, encoding="utf-8")
print(f"✓ {DATA_LOADER} actualizado")

# ── 2. Instalar pyarrow ──────────────────────────────────────────────────────
try:
    import pyarrow  # noqa: F401
    print(f"✓ pyarrow ya instalado ({pyarrow.__version__})")
except ImportError:
    print("Instalando pyarrow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "-q"])
    print("✓ pyarrow instalado")

# ── 3. Verificar carga del dataset ───────────────────────────────────────────
print("\n── Verificación de carga ──────────────────────────────────────────────")

import importlib, sys as _sys
for mod in list(_sys.modules.keys()):
    if "data_loader" in mod:
        del _sys.modules[mod]

_sys.path.insert(0, str(BASE))
from src.data_loader import auto_detect_and_load, filter_running_sessions  # noqa: E402

data_dir = BASE / "data" / "raw"
print(f"Directorio de datos: {data_dir}")
print(f"Archivos encontrados: {[f.name for f in data_dir.glob('*') if f.is_file()]}")

import logging
logging.basicConfig(level=logging.WARNING)

print("\nCargando dataset (puede tardar 1-2 min)...")
df_raw = auto_detect_and_load(str(data_dir))
print(f"\n  Sesiones totales : {len(df_raw):,}")
print(f"  Columnas         : {list(df_raw.columns)}")

df_run = filter_running_sessions(df_raw)
print(f"  Sesiones running : {len(df_run):,}")

if "userId" in df_run.columns:
    print(f"  Usuarios únicos  : {df_run['userId'].nunique():,}")
else:
    print("  AVISO: columna 'userId' no encontrada")

print("\n✓ Setup completado. Puedes ejecutar los notebooks en orden:")
print("   01_eda → 02_preprocessing → 04_modeling → 05_evaluation → ...")
