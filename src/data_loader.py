"""
data_loader.py — Carga del dataset FitRec (Ni et al., 2019).

FitRec está en formato JSONL: cada línea es un workout con campos:
  id, sport, gender, heart_rate (list), longitude (list), latitude (list),
  altitude (list), timestamp (list), url, userId.

Descarga: https://sites.google.com/eng.ucsd.edu/fitrec-project/home
  → FitRec.tar.gz (~4 GB descomprimido)
  → Archivo: workout_data.csv  (o fitrec.json / fitrec.jsonl según versión)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Campos de lista que vienen como secuencias por timestep
_LIST_FIELDS = ["heart_rate", "longitude", "latitude", "altitude", "timestamp"]

# Deporte objetivo (solo running)
TARGET_SPORT = "run"


def load_fitrec_jsonl(path: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Carga el archivo JSONL de FitRec y devuelve un DataFrame donde cada
    fila es una sesión, con las columnas de lista mantenidas como objetos Python.

    Parameters
    ----------
    path : str | Path
        Ruta al archivo .jsonl o .json (una sesión por línea).
    max_rows : int, optional
        Limita la carga a las primeras N filas (útil en desarrollo).

    Returns
    -------
    pd.DataFrame
        Columnas: id, sport, gender, userId + campos de lista.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {path}.\n"
            "Descárgalo desde https://sites.google.com/eng.ucsd.edu/fitrec-project/home\n"
            "y colócalo en data/raw/fitrec.jsonl"
        )

    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Línea %d malformada, se omite: %s", i, exc)

    df = pd.DataFrame(records)
    logger.info("Cargadas %d sesiones desde %s", len(df), path)
    return df


def load_fitrec_csv(path: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Alternativa: FitRec distribuido como CSV (una fila por sesión, listas
    almacenadas como strings JSON dentro de la celda).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {path}.\n"
            "Descárgalo desde https://sites.google.com/eng.ucsd.edu/fitrec-project/home"
        )

    df = pd.read_csv(path, nrows=max_rows)

    # Deserializar columnas que contienen listas como strings JSON
    for col in _LIST_FIELDS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(_safe_parse_list)

    logger.info("Cargadas %d sesiones desde CSV %s", len(df), path)
    return df


def _safe_parse_list(value):
    """Parsea una celda que puede ser un string JSON de lista o ya una lista."""
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
    """Filtra solo sesiones de running y elimina registros sin userId."""
    if "sport" in df.columns:
        df = df[df["sport"].str.lower() == TARGET_SPORT].copy()
    if "userId" in df.columns:
        df = df[df["userId"].notna()].copy()
    logger.info("Sesiones de running con userId válido: %d", len(df))
    return df


def auto_detect_and_load(data_dir: str | Path = "data/raw",
                         max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Detecta automáticamente el formato (jsonl / csv) en data_dir y carga.
    """
    data_dir = Path(data_dir)
    candidates_jsonl = list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.json"))
    candidates_csv = list(data_dir.glob("*.csv"))

    if candidates_jsonl:
        return load_fitrec_jsonl(candidates_jsonl[0], max_rows=max_rows)
    if candidates_csv:
        return load_fitrec_csv(candidates_csv[0], max_rows=max_rows)

    raise FileNotFoundError(
        f"No se encontró ningún archivo de datos en {data_dir}.\n"
        "Consulta README.md → Sección 'Dataset' para instrucciones de descarga."
    )
