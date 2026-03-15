"""
tests/test_spark_ingest.py
──────────────────────────
Tests for spark_ingest.py — built step by step.

Current step: 1 — SparkSession factory
"""

import pytest
from pyspark.sql import SparkSession
from ingestion.spark_ingest import build_spark


# ── Shared SparkSession fixture ───────────────────────────────────────────────
# scope="session" means pytest creates ONE SparkSession for the entire
# test run and reuses it across all tests.
#
# Why this matters:
#   Starting a SparkSession spins up a JVM — takes ~5 seconds.
#   If every test created its own, a 20-test suite = ~100s just on startup.
#   With scope="session": one 5s startup, then everything is near-instant.
#
# The yield pattern = pytest setup/teardown:
#   code before yield  →  runs once before all tests  (setup)
#   code after yield   →  runs once after all tests   (teardown)

@pytest.fixture(scope="session")
def spark():
    session = build_spark(app_name="test-suite")
    yield session
    session.stop()


# ── Step 1 tests: SparkSession factory ───────────────────────────────────────

class TestBuildSpark:

    def test_returns_spark_session(self, spark):
        """
        Most basic check: build_spark() returns a SparkSession.
        If this fails, the import or constructor is broken.
        """
        assert isinstance(spark, SparkSession)

    def test_session_is_active(self, spark):
        """
        Verifies the session is actually running.
        A stopped session raises RuntimeError on sparkContext access.
        """
        assert spark.sparkContext is not None

    def test_app_name_is_set(self, spark):
        """
        Confirms our custom app_name was passed through to Spark.
        You can see this name in the Spark UI at localhost:4040
        while a job is running.
        """
        assert spark.sparkContext.appName == "test-suite"

    def test_session_can_execute_job(self, spark):
        """
        Smoke test: the session can actually run a job.
        Creates a 3-row DataFrame and counts rows.
        If Spark's execution engine is broken this fails —
        rare, but good to catch before writing more code.
        """
        df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
        assert df.count() == 3

    def test_getorcreate_is_idempotent(self, spark):
        """
        getOrCreate() must return the SAME session, not a new one.
        Two active SparkSessions in the same process are not supported
        and cause subtle bugs. This test catches accidental duplication.
        """
        same_session = build_spark(app_name="different-name")
        assert same_session is spark