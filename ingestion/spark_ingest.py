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


# ── Step 4: normalise() ───────────────────────────────────────────────────────
# Cleans the raw DataFrame — no new columns, just fixing what's there.
#
# Three responsibilities:
#   1. Rename columns: remove spaces (breaks SQL and dbt)
#   2. Lowercase test_group: prevents "Ad" vs "ad" creating phantom groups
#   3. Fix hour=24: some exports use 1-based hours; normalise to 0-based
#
# We import functions as F — standard PySpark convention.
# F.col("name") selects a column, F.lower() lowercases it,
# F.trim() removes leading/trailing whitespace,
# F.when().otherwise() is PySpark's if/else for column expressions.

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def normalise(df: DataFrame) -> DataFrame:
    """
    Cleans and standardises the raw DataFrame.

    Input:  raw columns with spaces, mixed case, possible hour=24
    Output: snake_case columns, lowercase test_group, hours in 0-23
    """
    return (
        df
        # Step 4a: rename all columns to snake_case
        # toDF() takes a list of new names in the same order as current columns
        .toDF("user_id", "test_group", "converted",
              "total_ads", "most_ads_day", "most_ads_hour")

        # Step 4b: lowercase and trim test_group
        # F.trim() removes accidental whitespace, F.lower() normalises case
        .withColumn("test_group", F.trim(F.lower(F.col("test_group"))))

        # Step 4c: normalise most_ads_day capitalisation
        # "monday" → "Monday" so string comparisons work consistently
        .withColumn("most_ads_day", F.trim(F.initcap(F.col("most_ads_day"))))

        # Step 4d: fix hour=24 → hour=0
        # F.when(condition, value).otherwise(other_value) is PySpark's
        # equivalent of: IF condition THEN value ELSE other_value
        .withColumn(
            "most_ads_hour",
            F.when(F.col("most_ads_hour") == 24, F.lit(0))
             .otherwise(F.col("most_ads_hour"))
        )
    )
    
    
# ── Step 5: enrich() ──────────────────────────────────────────────────────────
# Adds derived columns that don't exist in the raw data.
# None of these change existing values — they only add new information.
#
# Why add these here rather than in dbt?
#   - is_control and ad_freq_bucket are needed by the Parquet writer
#     (Step 6) for partitioning decisions
#   - Having them in the raw Parquet means dbt models can use them
#     directly without re-deriving them
#   - ingestion_date is a pipeline metadata column — it belongs at
#     ingestion time, not in the transformation layer

from datetime import datetime, timezone


def enrich(df: DataFrame) -> DataFrame:
    """
    Adds derived columns to the normalised DataFrame.

    New columns:
        is_control     : True if test_group == 'psa' (control group)
        ad_freq_bucket : categorical bucket based on total_ads volume
        ingestion_date : date this pipeline run executed (for partitioning)
    """
    # Capture the current UTC date once — we use F.lit() to broadcast
    # this single value to every row. Using F.current_date() would also
    # work but F.lit() makes the value explicit and easier to mock in tests.
    today = datetime.now(tz=timezone.utc).date().isoformat()

    return (
        df

        # is_control: True for PSA (control), False for ad (treatment)
        # Cleaner than string comparison in every downstream query
        .withColumn(
            "is_control",
            F.col("test_group") == F.lit("psa")
        )

        # ad_freq_bucket: segments users by ad exposure volume
        # F.when() chains work like if / elif / elif / else
        # The order matters — first matching condition wins
        .withColumn(
            "ad_freq_bucket",
            F.when(F.col("total_ads") == 0,               F.lit("zero"))
             .when(F.col("total_ads").between(1, 5),       F.lit("low"))
             .when(F.col("total_ads").between(6, 20),      F.lit("medium"))
             .when(F.col("total_ads").between(21, 50),     F.lit("high"))
             .when(F.col("total_ads") > 50,                F.lit("very_high"))
             .otherwise(F.lit("unknown"))  # catches nulls
        )

        # ingestion_date: pipeline run date for Parquet partitioning
        # Stored as a date string — cast to DateType for proper partitioning
        .withColumn(
            "ingestion_date",
            F.to_date(F.lit(today))
        )
    )


# ── Step 6: write_parquet() ───────────────────────────────────────────────────
# Writes the enriched DataFrame to partitioned Parquet files.
#
# Key decisions:
#
#   partitionBy("ingestion_date")
#     Creates one folder per date: ingestion_date=2025-03-15/
#     Tomorrow's run overwrites only today's folder — past data is safe.
#     dbt incremental models can then filter WHERE ingestion_date = today.
#
#   mode("overwrite")
#     If today's partition already exists (e.g. pipeline re-run),
#     overwrite it cleanly. Without this, Spark appends and you get
#     duplicate rows on every re-run.
#
#   repartition(4)
#     Writes 4 Parquet files per partition — right-sized for ~600k rows.
#     Too many small files = slow reads (file open overhead).
#     Too few large files = can't parallelise reads.
#     Rule of thumb: aim for ~128MB per file.


def write_parquet(df: DataFrame, output_path: str) -> None:
    """
    Writes the enriched DataFrame to snappy-compressed Parquet,
    partitioned by ingestion_date.

    Args:
        df:          enriched DataFrame from enrich()
        output_path: directory to write to (e.g. "data/raw/")
    """
    log.info("Writing Parquet to: %s (partitioned by ingestion_date)",
             output_path)

    (
        df
        .repartition(4)
        .write
        .mode("overwrite")
        .partitionBy("ingestion_date")
        .parquet(output_path)
    )

    log.info("Write complete.")
    

# ── main() ────────────────────────────────────────────────────────────────────
# Wires all steps together into a single pipeline run.
# Called by `make ingest` and `make ingest-sample`.
#
# argparse lets us pass --output and --sample from the command line
# without hardcoding paths in the source code.

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Campaign A/B — ingestion job")
    p.add_argument("--output", default="data/raw/",
                   help="Output directory for Parquet files")
    p.add_argument("--sample", type=float, default=None,
                   help="Optional sample fraction for dev runs (e.g. 0.1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spark = build_spark()

    try:
        # Step 2: download
        csv_path = download_dataset()

        # Step 3: read
        df = read_raw(spark, csv_path)

        # Optional: sample for fast dev runs
        if args.sample:
            log.info("Sampling %.0f%% of rows", args.sample * 100)
            df = df.sample(fraction=args.sample, seed=42)

        # Step 4: normalise
        df = normalise(df)

        # Step 5: enrich
        df = enrich(df)

        # Step 6: write
        write_parquet(df, args.output)

        log.info("Pipeline complete. Rows written: %d", df.count())

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
