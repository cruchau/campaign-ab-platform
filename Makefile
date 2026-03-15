# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	python3 -m venv ~/.campaign_lift
	# source ~/.campaign_lift/bin/activate 


# ── Install ──────────────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
 

# ── Data download (kagglehub) ──────────────────────────────────────────────────
# Requires ~/.kaggle/kaggle.json API key.
# Downloads to ~/.cache/kagglehub/ and prints the path.
# Run this once — subsequent runs use the cache.
download:
	python -c "\
import kagglehub, pathlib; \
p = pathlib.Path(kagglehub.dataset_download('faviovaz/marketing-ab-testing')); \
print('Dataset cached at:', p)"


# ── Layer 0: Ingestion ─────────────────────────────────────────────────────────
# Download the CSV from Kaggle first:
#   kaggle datasets download -d faviovaz/marketing-ab-testing -p data/source/ --unzip
ingest:
	python ingestion/spark_ingest.py \
		--input  data/source/marketing_AB.csv \
		--output data/raw/
 
ingest-sample:
	python ingestion/spark_ingest.py \
		--input  data/source/marketing_AB.csv \
		--output data/raw/ \
		--sample 0.1
 

# ── Layer 1: dbt warehouse ─────────────────────────────────────────────────────
dbt:
	cd dbt_project && dbt run && dbt test
 
dbt-docs:
	cd dbt_project && dbt docs generate && dbt docs serve
 

# ── Layer 2+3: A/B engine + ML model ──────────────────────────────────────────
analyse:
	python ab_engine/stats.py
 
train:
	python ml/conversion_model.py
 

# ── Layer 4: Dashboard ─────────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py
 
# ── Tests & lint ───────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=ingestion --cov=ab_engine --cov=ml --cov-report=term-missing

test-step1:
	pytest tests/test_spark_ingest.py::TestBuildSpark \
	       tests/test_spark_ingest.py::TestDownloadDataset \
	       -v --tb=short

test-step2:
	pytest tests/test_spark_ingest.py::TestReadRaw -v --tb=short

test-step3:
	pytest tests/test_spark_ingest.py::TestNormalise -v --tb=short

test-step4:
	pytest tests/test_spark_ingest.py::TestEnrich -v --tb=short
 
lint:
	ruff check ingestion/ ab_engine/ ml/ dashboard/ tests/
 

# ── Full pipeline (CI equivalent) ─────────────────────────────────────────────
pipeline: ingest dbt analyse train
 

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	rm -rf data/raw data/processed
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true