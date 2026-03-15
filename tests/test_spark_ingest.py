"""
tests/test_spark_ingest.py
──────────────────────────
Tests for spark_ingest.py — built step by step.

Current step: 1 — SparkSession factory
"""

import pytest
from pathlib import Path
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


# ── Step 2 tests: dataset download ───────────────────────────────────────────
# We never call Kaggle for real in tests. Reasons:
#
#   1. SPEED: a network download in tests makes the suite slow and flaky
#   2. PORTABILITY: every developer needs a Kaggle key to run tests — bad
#   3. ISOLATION: tests should only test YOUR code, not Kaggle's servers
#
# Solution: unittest.mock.patch temporarily replaces
# kagglehub.dataset_download with a fake function we control.
# The real function is restored automatically after each test.
#
# This pattern is called "mocking at the boundary" — you mock the
# external call and test everything your code does around it.

from unittest.mock import patch, MagicMock
from ingestion.spark_ingest import download_dataset


class TestDownloadDataset:

    def test_returns_path_object(self, tmp_path):
        """
        download_dataset() must return a Path, not a raw string.
        Returning Path means callers can do path / "subdir" safely
        instead of string concatenation which breaks across OS.

        tmp_path is a pytest built-in fixture — it creates a fresh
        temporary directory for each test and deletes it afterwards.
        We use it as our fake Kaggle cache directory.
        """
        fake_csv = tmp_path / "marketing_AB.csv"
        fake_csv.touch()  # create an empty file at that path

        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   return_value=str(tmp_path)):
            result = download_dataset()

        assert isinstance(result, Path)

    def test_returns_csv_filename(self, tmp_path):
        """
        The returned Path must point to marketing_AB.csv specifically,
        not just the directory. Callers pass this directly to Spark's
        CSV reader so the filename must be correct.
        """
        fake_csv = tmp_path / "marketing_AB.csv"
        fake_csv.touch()

        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   return_value=str(tmp_path)):
            result = download_dataset()

        assert result.name == "marketing_AB.csv"

    def test_finds_csv_in_nested_subdirectory(self, tmp_path):
        """
        Kaggle sometimes nests files inside subdirectories in the zip.
        Our glob("**/*.csv") finds the file at any nesting depth.
        This test verifies that — without it a version bump on Kaggle's
        side could silently break the pipeline.
        """
        nested = tmp_path / "versions" / "1"
        nested.mkdir(parents=True)
        fake_csv = nested / "marketing_AB.csv"
        fake_csv.touch()

        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   return_value=str(tmp_path)):
            result = download_dataset()

        assert result.exists()

    def test_raises_file_not_found_if_csv_missing(self, tmp_path):
        """
        If the download directory exists but contains no CSV
        (e.g. Kaggle renamed the file), we raise FileNotFoundError
        with a clear message — not a cryptic AttributeError later
        when Spark tries to read a None path.

        pytest.raises() verifies both that the exception is raised
        AND that the message contains "marketing_AB.csv".
        """
        # tmp_path is an empty directory — no CSV inside
        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   return_value=str(tmp_path)):
            with pytest.raises(FileNotFoundError, match="marketing_AB.csv"):
                download_dataset()

    def test_calls_kaggle_with_correct_dataset_slug(self):
        """
        Verifies the exact dataset identifier passed to kagglehub.
        If someone changes the slug by accident, the download silently
        fetches the wrong dataset. This test catches that.

        MagicMock() creates a fake callable that records every call
        made to it. assert_called_once_with() then verifies it was
        called exactly once with the right argument.
        """
        mock_download = MagicMock(return_value="/nonexistent/path")

        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   mock_download):
            try:
                download_dataset()
            except FileNotFoundError:
                pass  # expected — /nonexistent/path doesn't exist

        mock_download.assert_called_once_with("faviovaz/marketing-ab-testing")      


# ── Step 3 tests: raw CSV reader ─────────────────────────────────────────────
# We test read_raw() without the real CSV file by creating a tiny
# in-memory DataFrame that mimics the CSV structure.
#
# Why not use the real CSV?
#   - The real file is 40MB — slow to read in every test run
#   - Tests should be fast and work without any external files
#   - We control exactly what data the tests see, making assertions precise
#
# spark.createDataFrame() lets us build a DataFrame from a plain
# Python list. We pair it with an explicit schema so the types match
# exactly what read_raw() would produce from the real CSV.

from ingestion.spark_ingest import read_raw, RAW_SCHEMA
from unittest.mock import patch
from pyspark.sql.types import StructType


class TestReadRaw:

    def _make_csv(self, tmp_path, content: str) -> Path:
        """
        Helper that writes a CSV string to a temp file and returns its Path.
        Used to give read_raw() a real file to read without needing Kaggle.
        """
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(content)
        return csv_file

    def test_returns_dataframe(self, spark, tmp_path):
        """
        read_raw() must return a Spark DataFrame, not None or a pandas df.
        Most basic smoke test — if this fails the function is broken entirely.
        """
        csv = self._make_csv(tmp_path, (
            "user id,test group,converted,total ads,most ads day,most ads hour\n"
            "1,ad,False,5,Monday,10\n"
        ))
        df = read_raw(spark, csv)
        from pyspark.sql import DataFrame
        assert isinstance(df, DataFrame)

    def test_correct_row_count(self, spark, tmp_path):
        """
        Verifies all rows are loaded — none silently dropped.
        3 data rows in, 3 rows out.
        """
        csv = self._make_csv(tmp_path, (
            "user id,test group,converted,total ads,most ads day,most ads hour\n"
            "1,ad,False,5,Monday,10\n"
            "2,psa,True,3,Tuesday,14\n"
            "3,ad,False,10,Friday,9\n"
        ))
        df = read_raw(spark, csv)
        assert df.count() == 3

    def test_correct_column_names(self, spark, tmp_path):
        """
        Verifies the DataFrame has exactly the columns we declared
        in RAW_SCHEMA — no extras, no missing, correct names including spaces.
        Column names with spaces are valid in Spark but must be handled
        carefully in SQL (use backticks: `user id`).
        """
        csv = self._make_csv(tmp_path, (
            "user id,test group,converted,total ads,most ads day,most ads hour\n"
            "1,ad,False,5,Monday,10\n"
        ))
        df = read_raw(spark, csv)
        assert df.columns == [
            "user id", "test group", "converted",
            "total ads", "most ads day", "most ads hour"
        ]

    def test_converted_is_boolean(self, spark, tmp_path):
        """
        The most important type check. If `converted` is read as StringType
        instead of BooleanType, then F.mean("converted") returns null
        instead of a conversion rate — a silent data quality bug.
        This test catches that before it reaches any dashboard.
        """
        csv = self._make_csv(tmp_path, (
            "user id,test group,converted,total ads,most ads day,most ads hour\n"
            "1,ad,False,5,Monday,10\n"
        ))
        df = read_raw(spark, csv)
        converted_type = dict(df.dtypes)["converted"]
        assert converted_type == "boolean", (
            f"Expected 'boolean' but got '{converted_type}' — "
            "conversion rate calculations will silently fail"
        )

    def test_nulls_in_nullable_columns(self, spark, tmp_path):
        """
        Verifies that empty values in nullable columns (total ads,
        most ads day, most ads hour) become null in Spark — not the
        string "null" or "None", which would break numeric aggregations.
        """
        csv = self._make_csv(tmp_path, (
            "user id,test group,converted,total ads,most ads day,most ads hour\n"
            "1,ad,False,,,\n"
        ))
        df = read_raw(spark, csv)
        row = df.collect()[0]
        assert row["total ads"] is None
        assert row["most ads day"] is None
        assert row["most ads hour"] is None  