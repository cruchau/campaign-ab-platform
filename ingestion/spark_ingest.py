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