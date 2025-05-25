import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "metrics"
DATASET_DIR = ROOT_DIR / "dataset"
BOOKS_DIR = DATASET_DIR / "books"
PARTS_DIR = DATASET_DIR / "parts"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
BOOKS_DIR.mkdir(parents=True, exist_ok=True)
PARTS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
