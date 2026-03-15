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
 
# Verify that the function was appended: grep -n "def " ingestion/spark_ingest.py
# grep stands for “Global Regular Expression Print”.
# It’s a command-line tool used to search text in files or output from other commands.
# It finds lines that match a pattern (text or regex) and prints them.
# Think of it as “find this text in a file or output”.

# ── Step 3: Raw CSV reader ────────────────────────────────────────────────────
# Why explicit schema instead of Spark's inferSchema=True?
#
#   1. PERFORMANCE: inferSchema reads the file twice — once to guess
#      types, once to load. Explicit schema = single pass.
#
#   2. CORRECTNESS: inference can silently misread types. If Spark reads
#      `converted` as StringType instead of BooleanType, every downstream
#      aggregation (conversion rate, lift %) silently produces wrong numbers.
#
#   3. FAIL FAST: mode="FAILFAST" makes Spark throw an error immediately
#      on any row that doesn't match the schema, instead of silently
#      filling nulls. You want to know about bad data at ingestion,
#      not discover it in a dashboard three days later.

from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Explicit schema matching the Kaggle CSV exactly.
# Column names match the CSV header including spaces —
# we rename them to snake_case in normalise() (Step 4).
RAW_SCHEMA = StructType([
    StructField("user id",       IntegerType(), nullable=False),
    StructField("test group",    StringType(),  nullable=False),
    StructField("converted",     BooleanType(), nullable=False),
    StructField("total ads",     IntegerType(), nullable=True),
    StructField("most ads day",  StringType(),  nullable=True),
    StructField("most ads hour", IntegerType(), nullable=True),
])


def read_raw(spark: SparkSession, csv_path: Path) -> "DataFrame":
    """
    Reads the raw CSV into a Spark DataFrame with enforced schema.

    mode=FAILFAST  → crash immediately on any malformed row
    nullValue=""   → treat empty strings as null
    header=True    → first row is column names, not data
    """
    log.info("Reading CSV: %s", csv_path)

    df = (
        spark.read
        .option("header", "true")
        .option("mode", "FAILFAST")
        .option("nullValue", "")
        .schema(RAW_SCHEMA)
        .csv(str(csv_path))
    )

    row_count = df.count()
    log.info("Loaded %d rows with %d columns", row_count, len(df.columns))
    return df