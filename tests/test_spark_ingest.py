"""
tests/test_spark_ingest.py
──────────────────────────
Tests for spark_ingest.py — built step by step.
"""

import pytest
from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    BooleanType, IntegerType, StringType,
    StructField, StructType, DateType
)

from ingestion.spark_ingest import (
    build_spark,
    download_dataset,
    read_raw,
    RAW_SCHEMA,
    normalise,
    enrich,
    write_parquet,
)


# ── Shared SparkSession fixture ───────────────────────────────────────────────
# scope="session" = one JVM for the entire test run.
# Without this, each test class spins up its own JVM (~5s each).
# With it: one 5s startup, everything else is near-instant.
#
# yield = pytest setup/teardown:
#   before yield → runs once before all tests  (setup)
#   after yield  → runs once after all tests   (teardown)

@pytest.fixture(scope="session")
def spark():
    session = build_spark(app_name="test-suite")
    yield session
    session.stop()


# ── Step 1: SparkSession factory ──────────────────────────────────────────────

class TestBuildSpark:

    def test_returns_spark_session(self, spark):
        """build_spark() must return a SparkSession, not None."""
        assert isinstance(spark, SparkSession)

    def test_session_is_active(self, spark):
        """A stopped session raises RuntimeError on sparkContext access."""
        assert spark.sparkContext is not None

    def test_app_name_is_set(self, spark):
        """Custom app_name must be passed through to Spark.
        Visible at localhost:4040 while a job runs."""
        assert spark.sparkContext.appName == "test-suite"

    def test_session_can_execute_job(self, spark):
        """Smoke test: session can run a real job.
        If Spark's execution engine is broken this fails."""
        df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
        assert df.count() == 3

    def test_getorcreate_is_idempotent(self, spark):
        """getOrCreate() must return the SAME session.
        Two active SparkSessions in one process cause subtle bugs."""
        same_session = build_spark(app_name="different-name")
        assert same_session is spark


# ── Step 2: Dataset download ──────────────────────────────────────────────────
# We never call Kaggle for real in tests:
#   - Network calls make tests slow and flaky
#   - Every developer would need a Kaggle API key
#   - Tests should only test YOUR code, not external services
#
# unittest.mock.patch temporarily replaces kagglehub.dataset_download
# with a fake we control. Real function restored after each test.
# Pattern: "mocking at the boundary"

class TestDownloadDataset:

    def test_returns_path_object(self, tmp_path):
        """Must return a Path, not a raw string.
        Path lets callers do path / 'subdir' safely across OS."""
        fake_csv = tmp_path / "marketing_AB.csv"
        fake_csv.touch()
        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   return_value=str(tmp_path)):
            result = download_dataset()
        assert isinstance(result, Path)

    def test_returns_csv_filename(self, tmp_path):
        """Returned Path must point to marketing_AB.csv specifically."""
        fake_csv = tmp_path / "marketing_AB.csv"
        fake_csv.touch()
        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   return_value=str(tmp_path)):
            result = download_dataset()
        assert result.name == "marketing_AB.csv"

    def test_finds_csv_in_nested_subdirectory(self, tmp_path):
        """Kaggle sometimes nests files in subdirectories.
        glob('**/*.csv') finds the file at any nesting depth."""
        nested = tmp_path / "versions" / "1"
        nested.mkdir(parents=True)
        fake_csv = nested / "marketing_AB.csv"
        fake_csv.touch()
        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   return_value=str(tmp_path)):
            result = download_dataset()
        assert result.exists()

    def test_raises_file_not_found_if_csv_missing(self, tmp_path):
        """Empty directory must raise FileNotFoundError with clear message.
        Better than a cryptic AttributeError when Spark reads None."""
        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   return_value=str(tmp_path)):
            with pytest.raises(FileNotFoundError, match="marketing_AB.csv"):
                download_dataset()

    def test_calls_kaggle_with_correct_dataset_slug(self):
        """Verifies the exact dataset slug passed to kagglehub.
        A wrong slug silently fetches the wrong dataset."""
        mock_download = MagicMock(return_value="/nonexistent/path")
        with patch("ingestion.spark_ingest.kagglehub.dataset_download",
                   mock_download):
            try:
                download_dataset()
            except FileNotFoundError:
                pass
        mock_download.assert_called_once_with("faviovaz/marketing-ab-testing")


# ── Step 3: Raw CSV reader ────────────────────────────────────────────────────
# Tests use tiny inline CSVs — not the real 40MB file.
# Benefits: fast, no Kaggle needed, full control over test data.
#
# Real CSV has 7 columns including a row_index prepended by Kaggle:
#   row_index, user id, test group, converted,
#   total ads, most ads day, most ads hour

class TestReadRaw:

    def _make_csv(self, tmp_path, content: str) -> Path:
        """Write a CSV string to a temp file and return its Path."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(content)
        return csv_file

    def _header(self):
        return (
            "row_index,user id,test group,converted,"
            "total ads,most ads day,most ads hour\n"
        )

    def test_returns_dataframe(self, spark, tmp_path):
        """read_raw() must return a Spark DataFrame."""
        csv = self._make_csv(tmp_path,
            self._header() +
            "0,1,ad,False,5,Monday,10\n"
        )
        df = read_raw(spark, csv)
        assert isinstance(df, DataFrame)

    def test_correct_row_count(self, spark, tmp_path):
        """All rows must be loaded — none silently dropped."""
        csv = self._make_csv(tmp_path,
            self._header() +
            "0,1,ad,False,5,Monday,10\n"
            "1,2,psa,True,3,Tuesday,14\n"
            "2,3,ad,False,10,Friday,9\n"
        )
        df = read_raw(spark, csv)
        assert df.count() == 3

    def test_correct_column_names(self, spark, tmp_path):
        """DataFrame must have exactly the 7 columns from RAW_SCHEMA."""
        csv = self._make_csv(tmp_path,
            self._header() +
            "0,1,ad,False,5,Monday,10\n"
        )
        df = read_raw(spark, csv)
        assert df.columns == [
            "row_index", "user id", "test group", "converted",
            "total ads", "most ads day", "most ads hour"
        ]

    def test_converted_is_boolean(self, spark, tmp_path):
        """converted must be BooleanType, not StringType.
        If wrong, F.mean('converted') returns null silently."""
        csv = self._make_csv(tmp_path,
            self._header() +
            "0,1,ad,False,5,Monday,10\n"
        )
        df = read_raw(spark, csv)
        assert dict(df.dtypes)["converted"] == "boolean"

    def test_nulls_in_nullable_columns(self, spark, tmp_path):
        """Empty values in nullable columns must become null.
        Not the string 'null' or 'None' — those break aggregations."""
        csv = self._make_csv(tmp_path,
            self._header() +
            "0,1,ad,False,,,\n"
        )
        df = read_raw(spark, csv)
        row = df.collect()[0]
        assert row["total ads"]     is None
        assert row["most ads day"]  is None
        assert row["most ads hour"] is None


# ── Step 4: normalise() ───────────────────────────────────────────────────────
# Tests each transformation independently on a minimal DataFrame.
# _raw_df() includes row_index to match real read_raw() output —
# normalise() renames and drops it via toDF().

class TestNormalise:

    def _raw_df(self, spark, test_group="ad", most_ads_day="monday",
                most_ads_hour=10, total_ads=5):
        """Minimal raw DataFrame matching read_raw() output."""
        schema = StructType([
            StructField("row_index",     IntegerType(), nullable=True),
            StructField("user id",       IntegerType(), nullable=False),
            StructField("test group",    StringType(),  nullable=False),
            StructField("converted",     BooleanType(), nullable=False),
            StructField("total ads",     IntegerType(), nullable=True),
            StructField("most ads day",  StringType(),  nullable=True),
            StructField("most ads hour", IntegerType(), nullable=True),
        ])
        return spark.createDataFrame(
            [(0, 1, test_group, False, total_ads, most_ads_day, most_ads_hour)],
            schema=schema
        )

    def test_renames_columns_to_snake_case(self, spark):
        """Spaces in column names break SQL and dbt. Must use underscores."""
        df = normalise(self._raw_df(spark))
        assert "user_id"       in df.columns
        assert "test_group"    in df.columns
        assert "total_ads"     in df.columns
        assert "most_ads_day"  in df.columns
        assert "most_ads_hour" in df.columns

    def test_removes_spaced_column_names(self, spark):
        """Old spaced names must be completely gone after rename."""
        df = normalise(self._raw_df(spark))
        assert "user id"    not in df.columns
        assert "test group" not in df.columns
        assert "total ads"  not in df.columns

    def test_drops_row_index(self, spark):
        """row_index is a Kaggle artefact — must be dropped."""
        df = normalise(self._raw_df(spark))
        assert "row_index" not in df.columns

    def test_lowercases_test_group(self, spark):
        """'Ad', 'AD', 'aD' must all become 'ad'.
        Mixed case silently creates extra groups in GROUP BY."""
        df = normalise(self._raw_df(spark, test_group="Ad"))
        assert df.collect()[0]["test_group"] == "ad"

    def test_trims_whitespace_from_test_group(self, spark):
        """'  ad  ' must become 'ad'. Whitespace breaks equality checks."""
        df = normalise(self._raw_df(spark, test_group="  ad  "))
        assert df.collect()[0]["test_group"] == "ad"

    def test_title_cases_most_ads_day(self, spark):
        """'monday' → 'Monday'. Consistent case for downstream filters."""
        df = normalise(self._raw_df(spark, most_ads_day="monday"))
        assert df.collect()[0]["most_ads_day"] == "Monday"

    def test_fixes_hour_24_to_0(self, spark):
        """Some exports use hour=24 for midnight. Normalise to 0."""
        df = normalise(self._raw_df(spark, most_ads_hour=24))
        assert df.collect()[0]["most_ads_hour"] == 0

    def test_leaves_valid_hours_unchanged(self, spark):
        """Hour fix must only change 24 — not any other value."""
        df = normalise(self._raw_df(spark, most_ads_hour=13))
        assert df.collect()[0]["most_ads_hour"] == 13


# ── Step 5: enrich() ──────────────────────────────────────────────────────────
# Tests boundary values of each bucket — not just the middle.
# Boundary bugs (is 50 "high" or "very_high"?) are the most common
# source of silent errors in bucketing logic.

class TestEnrich:

    def _normalised_df(self, spark, total_ads=5, test_group="ad"):
        """Minimal normalised DataFrame — what normalise() produces."""
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
        df = enrich(self._normalised_df(spark))
        assert "is_control" in df.columns

    def test_is_control_true_for_psa(self, spark):
        df = enrich(self._normalised_df(spark, test_group="psa"))
        assert df.collect()[0]["is_control"] is True

    def test_is_control_false_for_ad(self, spark):
        df = enrich(self._normalised_df(spark, test_group="ad"))
        assert df.collect()[0]["is_control"] is False

    def test_adds_ad_freq_bucket_column(self, spark):
        df = enrich(self._normalised_df(spark))
        assert "ad_freq_bucket" in df.columns

    def test_bucket_zero(self, spark):
        df = enrich(self._normalised_df(spark, total_ads=0))
        assert df.collect()[0]["ad_freq_bucket"] == "zero"

    def test_bucket_low_boundary_start(self, spark):
        """1 ad = lower boundary of 'low'."""
        df = enrich(self._normalised_df(spark, total_ads=1))
        assert df.collect()[0]["ad_freq_bucket"] == "low"

    def test_bucket_low_boundary_end(self, spark):
        """5 ads = upper boundary of 'low'."""
        df = enrich(self._normalised_df(spark, total_ads=5))
        assert df.collect()[0]["ad_freq_bucket"] == "low"

    def test_bucket_medium(self, spark):
        """6 ads = first value above 'low'."""
        df = enrich(self._normalised_df(spark, total_ads=6))
        assert df.collect()[0]["ad_freq_bucket"] == "medium"

    def test_bucket_high_boundary(self, spark):
        """50 ads = upper boundary of 'high'."""
        df = enrich(self._normalised_df(spark, total_ads=50))
        assert df.collect()[0]["ad_freq_bucket"] == "high"

    def test_bucket_very_high(self, spark):
        """51 ads = first value above 'high'."""
        df = enrich(self._normalised_df(spark, total_ads=51))
        assert df.collect()[0]["ad_freq_bucket"] == "very_high"

    def test_bucket_null_ads_is_unknown(self, spark):
        """Null total_ads must become 'unknown'.
        Without this, nulls produce None and break GROUP BY."""
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
        assert enrich(df).collect()[0]["ad_freq_bucket"] == "unknown"

    def test_adds_ingestion_date_column(self, spark):
        df = enrich(self._normalised_df(spark))
        assert "ingestion_date" in df.columns

    def test_ingestion_date_is_today(self, spark):
        df = enrich(self._normalised_df(spark))
        assert df.collect()[0]["ingestion_date"] == date.today()


# ── Step 6: write_parquet() + end-to-end ──────────────────────────────────────

class TestWriteParquet:

    def _enriched_df(self, spark):
        """Minimal enriched DataFrame — what enrich() produces."""
        schema = StructType([
            StructField("user_id",        IntegerType(), nullable=False),
            StructField("test_group",     StringType(),  nullable=False),
            StructField("converted",      BooleanType(), nullable=False),
            StructField("total_ads",      IntegerType(), nullable=True),
            StructField("most_ads_day",   StringType(),  nullable=True),
            StructField("most_ads_hour",  IntegerType(), nullable=True),
            StructField("is_control",     BooleanType(), nullable=False),
            StructField("ad_freq_bucket", StringType(),  nullable=False),
            StructField("ingestion_date", DateType(),    nullable=False),
        ])
        today = date.today()
        return spark.createDataFrame(
            [(1, "ad",  False, 5, "Monday",  10, False, "low", today),
             (2, "psa", True,  3, "Tuesday", 14, True,  "low", today)],
            schema=schema
        )

    def test_creates_output_directory(self, spark, tmp_path):
        """Output directory must be created automatically."""
        output = str(tmp_path / "output")
        write_parquet(self._enriched_df(spark), output)
        assert (tmp_path / "output").exists()

    def test_creates_parquet_files(self, spark, tmp_path):
        """Output must contain actual .parquet files."""
        output = str(tmp_path / "output")
        write_parquet(self._enriched_df(spark), output)
        assert len(list((tmp_path / "output").rglob("*.parquet"))) > 0

    def test_partitioned_by_ingestion_date(self, spark, tmp_path):
        """Must create ingestion_date=YYYY-MM-DD/ partition folders."""
        output = str(tmp_path / "output")
        write_parquet(self._enriched_df(spark), output)
        partition_dirs = [
            d for d in (tmp_path / "output").iterdir()
            if d.is_dir() and d.name.startswith("ingestion_date=")
        ]
        assert len(partition_dirs) > 0

    def test_data_readable_after_write(self, spark, tmp_path):
        """Data written must be readable back with correct row count."""
        output = str(tmp_path / "output")
        original = self._enriched_df(spark)
        write_parquet(original, output)
        assert spark.read.parquet(output).count() == original.count()


class TestEndToEnd:

    def test_full_pipeline_on_small_dataframe(self, spark, tmp_path):
        """
        Runs read_raw → normalise → enrich → write_parquet → read back.
        Catches bugs that only appear when functions are composed together.
        Uses an inline CSV — no Kaggle needed, runs in milliseconds.
        """
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "row_index,user id,test group,converted,"
            "total ads,most ads day,most ads hour\n"
            "0,1,ad,False,5,Monday,10\n"
            "1,2,psa,True,3,Tuesday,14\n"
            "2,3,ad,False,0,Wednesday,9\n"
            "3,4,psa,False,25,Thursday,18\n"
            "4,5,ad,True,100,Friday,21\n"
        )
        output = str(tmp_path / "output")

        df = read_raw(spark, csv_file)
        df = normalise(df)
        df = enrich(df)
        write_parquet(df, output)

        result = spark.read.parquet(output)
        rows = {r["user_id"]: r for r in result.collect()}

        assert result.count() == 5
        assert "is_control"     in result.columns
        assert "ad_freq_bucket" in result.columns
        assert "ingestion_date" in result.columns
        assert "user_id"        in result.columns
        assert "user id"        not in result.columns
        assert "row_index"      not in result.columns

        assert rows[1]["ad_freq_bucket"] == "low"        # 5 ads
        assert rows[3]["ad_freq_bucket"] == "zero"       # 0 ads
        assert rows[4]["ad_freq_bucket"] == "high"       # 25 ads
        assert rows[5]["ad_freq_bucket"] == "very_high"  # 100 ads