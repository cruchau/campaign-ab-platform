"""
ingestion/spark_ingest.py

Layer 0 - ingestion Job for the campaign A/B test. 

This job will read the data from the source, transform it, and write it to the destination.    
"""

import logging 
import sys 

from pyspark.sql import SparkSession

# Logging 
# basicCongig sets up a single logger for the whole module.
# Every function below uses `log.info(...)` instead of print() because:
#   - logs include a timestamp (critical for debugging pipeline runs)
#   - log level can be changed without touching code (INFO in dev, WARN in prod)
#   - CI captures structured log output cleanly

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("spark_ingest")


# Step 1: SparkSession factory
# Why a factory function rather than a global SparkSession object?
#
#   1. TESTABILITY: tests call build_spark() and get a fresh session
#      without importing a global that might already be stopped.
#
#   2. CONFIGURABILITY: in production you swap master("local[*]")
#      for a cluster URL — no other code changes needed.
#
#   3. SINGLE RESPONSIBILITY: all Spark config lives here, nowhere else.
#
# Config decisions explained:
#   local[*]                 use all CPU cores on this machine
#   shuffle.partitions = 8   Spark default is 200 — far too many for
#                            ~600k rows; 8 is right-sized here
#   compression = snappy     best balance of speed vs file size for Parquet
#   timeZone = UTC           prevents silent timezone bugs on timestamp cols
#   driver.memory = 2g       prevents OOM errors on a laptop

def build_spark(app_name: str = 'campaign-ab-platform') -> SparkSession:
    """
    Factory function to create a SparkSession with appropriate config.
    
    Use local[*] master for dev/CI.
    
    In production, pass master= as an argument read from an env var.
    """
    log.info("Building SparkSession: %s", app_name)
    
    session = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    
    # Suppress Spark's own verbose INFO logs so ours stay readable.
    # WARN still shows — useful for partition skew / plan warnings.
    session.sparkContext.setLogLevel("WARN")

    log.info("SparkSession started (Spark %s)", session.version)
    return session

# Step 2: Dataset Download
# Why Kagglehub instead of manual download?
#
#   1. REPRODUCIBLE: anyone who clones the repo gets the data automatically.
#      No "download this file and put it here" steps that people skip.
#
#   2. CACHED: kagglehub stores data in ~/.cache/kagglehub/ after the first
#      download. Re-running the pipeline doesn't re-download 40MB each time.
#
#   3. VERSIONED: pinned to "latest version" by default. You can lock to
#      a specific version for full reproducibility in production.

import kagglehub
from pathlib import Path

KAGGLE_DATASET = "faviovaz/marketing-ab-testing"
CSV_FILENAME   = "marketing_AB.csv"

def download_dataset() -> Path:
    """
    Downloads the marketing A/B dataset from Kaggle via kagglehub.
    Returns the Path to the CSV file (from cache if already downloaded).

    kagglehub.dataset_download() returns the directory containing the
    dataset files — we search inside it for our specific CSV.
    """
    log.info("Fetching dataset: %s", KAGGLE_DATASET)

    # First run  → downloads to ~/.cache/kagglehub/datasets/faviovaz/...
    # Later runs → returns cached path instantly, no network call
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    log.info("Dataset directory: %s", dataset_dir)

    # Search recursively in case Kaggle nests files inside subdirectories.
    # We search rather than hardcode the path in case the internal zip
    # structure changes between dataset versions.
    matches = list(dataset_dir.glob(f"**/{CSV_FILENAME}"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {CSV_FILENAME} in {dataset_dir}. "
            f"Files present: {list(dataset_dir.iterdir())}"
        )

    csv_path = matches[0]
    log.info("CSV located at: %s", csv_path)
    return csv_path