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



# ── Step 4 tests: normalise() ─────────────────────────────────────────────────
# We test each transformation independently.
# Each test creates a minimal 1-row DataFrame with only the columns
# relevant to that test — keeps tests focused and failures easy to diagnose.
#
# spark.createDataFrame(data, schema) where schema is a list of strings
# is a shorthand for simple schemas — Spark infers types from the data.
# For precise type control we use the full StructType (like in read_raw).

from ingestion.spark_ingest import normalise
from pyspark.sql.types import (
    BooleanType, IntegerType, StringType, StructField, StructType
)


class TestNormalise:

    def _raw_df(self, spark, test_group="ad", most_ads_day="monday",
                most_ads_hour=10, total_ads=5):
        """
        Helper that creates a minimal raw DataFrame matching what
        read_raw() would produce — spaces in column names, raw values.
        We reuse this across tests, overriding only what each test needs.
        """
        schema = StructType([
            StructField("user id",       IntegerType(), nullable=False),
            StructField("test group",    StringType(),  nullable=False),
            StructField("converted",     BooleanType(), nullable=False),
            StructField("total ads",     IntegerType(), nullable=True),
            StructField("most ads day",  StringType(),  nullable=True),
            StructField("most ads hour", IntegerType(), nullable=True),
        ])
        return spark.createDataFrame(
            [(1, test_group, False, total_ads, most_ads_day, most_ads_hour)],
            schema=schema
        )

    def test_renames_columns_to_snake_case(self, spark):
        """
        Spaces in column names break SQL queries and dbt models.
        After normalise(), all column names must use underscores.
        """
        df = normalise(self._raw_df(spark))
        assert "user_id"       in df.columns
        assert "test_group"    in df.columns
        assert "total_ads"     in df.columns
        assert "most_ads_day"  in df.columns
        assert "most_ads_hour" in df.columns

    def test_removes_spaced_column_names(self, spark):
        """
        The old spaced names must be completely gone — not just
        shadowed by the new ones. Having both would confuse dbt.
        """
        df = normalise(self._raw_df(spark))
        assert "user id"       not in df.columns
        assert "test group"    not in df.columns
        assert "total ads"     not in df.columns

    def test_lowercases_test_group(self, spark):
        """
        "Ad", "AD", "aD" must all become "ad".
        Without this, groupBy("test_group") silently creates
        extra groups and the A/B split looks wrong.
        """
        df = normalise(self._raw_df(spark, test_group="Ad"))
        assert df.collect()[0]["test_group"] == "ad"

    def test_trims_whitespace_from_test_group(self, spark):
        """
        "  ad  " must become "ad".
        Trailing whitespace is invisible in logs but breaks equality checks.
        """
        df = normalise(self._raw_df(spark, test_group="  ad  "))
        assert df.collect()[0]["test_group"] == "ad"

    def test_title_cases_most_ads_day(self, spark):
        """
        "monday" → "Monday", "FRIDAY" → "Friday".
        Consistent capitalisation so downstream filters like
        WHERE most_ads_day = 'Monday' work reliably.
        """
        df = normalise(self._raw_df(spark, most_ads_day="monday"))
        assert df.collect()[0]["most_ads_day"] == "Monday"

    def test_fixes_hour_24_to_0(self, spark):
        """
        Some data exports use hour=24 for midnight instead of 0.
        Both mean the same time but 24 is out of the 0-23 range
        and breaks time-of-day analysis. We normalise it to 0.
        """
        df = normalise(self._raw_df(spark, most_ads_hour=24))
        assert df.collect()[0]["most_ads_hour"] == 0

    def test_leaves_valid_hours_unchanged(self, spark):
        """
        The hour=24 fix must only change 24 — not any other value.
        Without this test, a bug could zero out all hours.
        """
        df = normalise(self._raw_df(spark, most_ads_hour=13))
        assert df.collect()[0]["most_ads_hour"] == 13
        

# ── Step 5 tests: enrich() ────────────────────────────────────────────────────
# We test each derived column independently.
# Key principle: test the BOUNDARY values of each bucket, not just
# the middle. Boundary bugs (is 50 "high" or "very_high"?) are the
# most common source of silent data errors in bucketing logic.

from ingestion.spark_ingest import enrich
from datetime import date


class TestEnrich:

    def _normalised_df(self, spark, total_ads=5, test_group="ad"):
        """
        Helper that creates a minimal normalised DataFrame —
        what normalise() would produce — for enrich() to work on.
        """
        schema = StructType([
            StructField("user_id",       IntegerType(), nullable=False),
            StructField("test_group",    StringType(),  nullable=False),
            StructField("converted",     BooleanType(), nullable=False),
            StructField("total_ads",     IntegerType(), nullable=True),
            StructField("most_ads_day",  StringType(),  nullable=True),
            StructField("most_ads_hour", IntegerType(), nullable=True),
        ])
        return spark.createDataFrame(
            [(1, test_group, False, total_ads, "Monday", 10)],
            schema=schema
        )

    def test_adds_is_control_column(self, spark):
        """
        enrich() must add an is_control column.
        Existence check — type and value checked in separate tests.
        """
        df = enrich(self._normalised_df(spark))
        assert "is_control" in df.columns

    def test_is_control_true_for_psa(self, spark):
        """
        PSA group = control group → is_control must be True.
        """
        df = enrich(self._normalised_df(spark, test_group="psa"))
        assert df.collect()[0]["is_control"] is True

    def test_is_control_false_for_ad(self, spark):
        """
        Ad group = treatment group → is_control must be False.
        """
        df = enrich(self._normalised_df(spark, test_group="ad"))
        assert df.collect()[0]["is_control"] is False

    def test_adds_ad_freq_bucket_column(self, spark):
        """
        enrich() must add an ad_freq_bucket column.
        """
        df = enrich(self._normalised_df(spark))
        assert "ad_freq_bucket" in df.columns

    def test_bucket_zero(self, spark):
        """0 ads → 'zero' bucket."""
        df = enrich(self._normalised_df(spark, total_ads=0))
        assert df.collect()[0]["ad_freq_bucket"] == "zero"

    def test_bucket_low_boundary_start(self, spark):
        """1 ad → 'low' bucket — lower boundary of low."""
        df = enrich(self._normalised_df(spark, total_ads=1))
        assert df.collect()[0]["ad_freq_bucket"] == "low"

    def test_bucket_low_boundary_end(self, spark):
        """5 ads → still 'low' — upper boundary of low."""
        df = enrich(self._normalised_df(spark, total_ads=5))
        assert df.collect()[0]["ad_freq_bucket"] == "low"

    def test_bucket_medium(self, spark):
        """6 ads → 'medium' — first value above low."""
        df = enrich(self._normalised_df(spark, total_ads=6))
        assert df.collect()[0]["ad_freq_bucket"] == "medium"

    def test_bucket_high_boundary(self, spark):
        """50 ads → 'high' — upper boundary of high."""
        df = enrich(self._normalised_df(spark, total_ads=50))
        assert df.collect()[0]["ad_freq_bucket"] == "high"

    def test_bucket_very_high(self, spark):
        """51 ads → 'very_high' — first value above high."""
        df = enrich(self._normalised_df(spark, total_ads=51))
        assert df.collect()[0]["ad_freq_bucket"] == "very_high"

    def test_bucket_null_ads_is_unknown(self, spark):
        """
        Null total_ads → 'unknown' bucket.
        Without this, nulls fall through all conditions silently
        and produce None in the bucket column, breaking GROUP BY.
        """
        schema = StructType([
            StructField("user_id",       IntegerType(), nullable=False),
            StructField("test_group",    StringType(),  nullable=False),
            StructField("converted",     BooleanType(), nullable=False),
            StructField("total_ads",     IntegerType(), nullable=True),
            StructField("most_ads_day",  StringType(),  nullable=True),
            StructField("most_ads_hour", IntegerType(), nullable=True),
        ])
        df = spark.createDataFrame(
            [(1, "ad", False, None, "Monday", 10)],
            schema=schema
        )
        result = enrich(df).collect()[0]["ad_freq_bucket"]
        assert result == "unknown"

    def test_adds_ingestion_date_column(self, spark):
        """
        enrich() must add an ingestion_date column.
        Used for Parquet partitioning in Step 6.
        """
        df = enrich(self._normalised_df(spark))
        assert "ingestion_date" in df.columns

    def test_ingestion_date_is_today(self, spark):
        """
        ingestion_date must equal today's UTC date.
        We compare against date.today() — if this test runs
        exactly at midnight UTC it could theoretically fail,
        but that's acceptable for a pipeline test.
        """
        df = enrich(self._normalised_df(spark))
        result = df.collect()[0]["ingestion_date"]
        assert result == date.today()