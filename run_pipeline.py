"""
run_pipeline.py — Ejecuta todos los notebooks en orden y guarda los resultados.

Uso:
    python run_pipeline.py                  # ejecuta todos
    python run_pipeline.py --only 01 02    # ejecuta solo los indicados
    python run_pipeline.py --skip 06 07    # salta los indicados

Requisitos:
    pip install nbconvert nbformat jupyter
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
OUTPUT_DIR = Path(__file__).parent / "notebooks" / "executed"

# Orden de ejecución
PIPELINE = [
    "00_pipeline.ipynb",
    "01_eda.ipynb",
    "02_preprocessing.ipynb",
    "03_feature_engineering.ipynb",
    "04_modeling.ipynb",
    "05_evaluation.ipynb",
    "06_shap_interpretability.ipynb",
    "07_acwr_analysis.ipynb",
]


def run_notebook(nb_path: Path, output_dir: Path) -> bool:
    output_path = output_dir / nb_path.name
    print(f"\n{'='*60}")
    print(f"Ejecutando: {nb_path.name}")
    print(f"Guardando en: {output_path}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=3600",  # 1h máximo por notebook
            "--ExecutePreprocessor.kernel_name=python3",
            "--output", str(output_path),
            str(nb_path),
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"OK  {nb_path.name} — {elapsed:.0f}s")
        return True
    else:
        print(f"ERROR {nb_path.name} — {elapsed:.0f}s")
        print(result.stderr[-3000:] if result.stderr else "(sin stderr)")
        return False


def ensure_nbconvert():
    try:
        import nbconvert  # noqa: F401
    except ImportError:
        print("Instalando jupyter + nbconvert...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nbconvert", "nbformat", "jupyter", "-q"])
        print("Instalado.")


def main():
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline de notebooks.")
    parser.add_argument("--only", nargs="*", help="Prefijos de notebooks a ejecutar (ej: 01 02)")
    parser.add_argument("--skip", nargs="*", help="Prefijos de notebooks a saltar (ej: 06 07)")
    args = parser.parse_args()

    ensure_nbconvert()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = [NOTEBOOKS_DIR / nb for nb in PIPELINE if (NOTEBOOKS_DIR / nb).exists()]

    if args.only:
        notebooks = [nb for nb in notebooks if any(nb.name.startswith(p) for p in args.only)]
    if args.skip:
        notebooks = [nb for nb in notebooks if not any(nb.name.startswith(p) for p in args.skip)]

    if not notebooks:
        print("No hay notebooks que ejecutar con los filtros indicados.")
        return

    print(f"Ejecutando {len(notebooks)} notebooks → resultados en {OUTPUT_DIR}")

    results = {}
    total_start = time.time()
    for nb in notebooks:
        ok = run_notebook(nb, OUTPUT_DIR)
        results[nb.name] = ok
        if not ok:
            print(f"\nPipeline detenido en {nb.name}. Revisa el error arriba.")
            break

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"RESUMEN — {total/60:.1f} min totales")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "OK " if ok else "ERR"
        print(f"  {status}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
