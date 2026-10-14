Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
--------------------------------------------------------------------------------------------------------------------------------------------

# jul26_bmlops_int_rakuten — Phase 1

Classification multiclasse des produits Rakuten (`prdtypecode`, ~27 classes) à
partir du texte (`designation`, `description`) et de l'image du produit.
Baseline **multimodal** : TF-IDF (texte) + embeddings ResNet18 gelé (image),
fusionnés dans un pipeline scikit-learn, classifieur Régression Logistique.

## 1. Prerequisites

- Docker and Docker Compose
- Raw data files from the Rakuten challenge (see §3)

## 2. Configuration (`.env`)

Créer un fichier `.env` à la racine du projet (voir `.env.example` si présent) :

```env
POSTGRES_USER=<ton_user>
POSTGRES_PASSWORD=<ton_password>
POSTGRES_DB=<ta_db>
POSTGRES_HOST=db
POSTGRES_PORT=5432

MODEL_NAME=baseline_logreg
TARGET_COLUMN=prdtypecode
TEST_SIZE=0.2
RANDOM_STATE=42

API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# --- MLflow ---
MLFLOW_TRACKING_URI=postgresql+psycopg://"POSTGRES_USER":"POSTGRES_PASSWORD"@localhost:5432/mlops
# replace "POSTGRES_PASSWORD" and "POSTGRES_PASSWORD"
MLFLOW_EXPERIMENT_NAME=rakuten-baseline-v2
MLFLOW_REGISTERED_MODEL_NAME=rakuten_baseline

# --- Airflow ---
AIRFLOW_UID=1000
AIRFLOW_FERNET_KEY=<create a key>
AIRFLOW_JWT_SECRET=<create a key>
```

> ⚠️ **Important** : "POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB" 
are only read by Postgres during the very first initialization of the data volume.
If you change them afterwards, Postgres will ignore the new values and keep the old ones → authentication errors.
In that case, delete the volume (§7 Troubleshooting) to force a clean reinitialization.

## 3. Raw data

The raw files are versioned with DVC and stored on DagsHub. You no longer need
to obtain the Rakuten dataset by hand.

### Credentials

The remote is S3-compatible storage hosted by DagsHub, so DVC has to
authenticate. Each person uses **their own** token - they are not shared, and
somebody else's will not work for you.

1. Create a free account at https://dagshub.com if you do not have one.
2. Open https://dagshub.com/SrToshi/rakuten-data and click the green **Data**
   button. The panel shows the two commands below with your token already
   filled in, ready to copy.
   (Alternatively, generate one at https://dagshub.com/user/settings/tokens.)
3. Run them from the repository root, after installing DVC (next section).

```bash
dvc remote modify --local origin access_key_id     <YOUR_DAGSHUB_TOKEN>
dvc remote modify --local origin secret_access_key <YOUR_DAGSHUB_TOKEN>
```

DagsHub uses the same value as both key and secret. That is not a typo.

`--local` is what keeps the token out of git: it writes to `.dvc/config.local`,
which DVC adds to `.dvc/.gitignore` automatically. The shared `.dvc/config`
carries only the remote address, and that is the one that gets committed.

**Never put a token in a commit, a Slack message, or a screenshot.** A token
that has been seen by anyone else has to be treated as compromised and
regenerated - which breaks everybody else's setup until they update theirs.

If `dvc pull` returns a 403, your token is missing, mistyped, or your account
lacks access to the repository.

### Installing DVC

DVC runs outside the containers, so it needs a local Python environment.
Everything else in this project runs in Docker; this is the one exception.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS:  source .venv/bin/activate
pip install -e ".[dev]"         # installs dvc[s3] among the dev dependencies
```

DVC is declared under `[dev]`, not in `requirements.txt`, on purpose. It is a
development and CI tool: by the time the API, Airflow or Streamlit run, the
data is already in PostgreSQL. The API image has no reason to carry it.

### Getting the data

```bash
dvc pull
```

| File | Size |
|---|---|
| `X_train_update.csv` | 54 MB |
| `Y_train_CVw08PX.csv` | 0.9 MB |
| `X_test_update.csv` | 8.8 MB |
| `image_train/image_train/` | ~85k images, several GB |

**If you already have the data on disk, you probably do not want a full
`dvc pull`.** Your DVC cache starts empty, so DVC will download everything
again and keep a second copy under `.dvc/cache` - several gigabytes, for files
you already have. To check that the remote is reachable without paying for
that, pull only a small one:

```bash
dvc pull src/data/raw/X_train_update.csv.dvc
```

`dvc status` tells you what DVC thinks is missing. `dvc status --cloud`
compares your cache against the remote without downloading anything.

### Where DVC sits, and where it does not

Each tool has one job, and the boundaries are deliberate:

| Tool | Versions |
|---|---|
| git | code, and the small `.dvc` pointer files |
| DVC | the raw data files |
| MLflow Model Registry | models, with stages and promotion |
| Airflow | orchestration |
| PostgreSQL | a working copy of the data, rebuildable from the above |

**DVC does not version the database.** It versions files, and a Postgres table
is not a file. What it versions are the source files the ingestion reads, so
that the database can always be rebuilt from a known starting point.

**There is no `dvc.yaml` and no `dvc repro`.** DVC can orchestrate pipelines,
but Airflow already does that here, and two orchestrators drift apart. DVC is
used for storage and provenance only.

**Models are not in DVC.** The Model Registry already versions them, with
stages and promotion. Two sources of truth for "which model is the good one"
would eventually disagree.

### Data lineage

Every row in `raw_samples` points at the batch it was ingested in, and each
batch records the DVC hash of every file it came from, plus the git commit that
held those pointers. Training logs its batches to MLflow. So the chain reads:

```
model in Production → MLflow run → data_batches row → md5 → dvc pull → the exact bytes
```

A hash here is an address, not a label: `dvc pull` resolves it back to the
data. Inspect it with:

```sql
SELECT id, sources, git_commit, n_rows, note, ingested_at FROM data_batches;
```

### What to expect the first time you pull this branch

1. **The database schema changes.** `raw_samples` gains a nullable `batch_id`
   column, and a `data_batches` table appears. `init_db()` applies this
   automatically at API startup, so `docker compose up -d --build api` is
   enough. Nothing existing is dropped or rewritten.

2. **Your existing rows will have `batch_id = NULL`**, because they were
   ingested before any of this existed. Attach them to a batch with:

```bash
   python scripts/backfill_data_batches.py
```

   Safe to run more than once: it does nothing when no orphan rows are left.
   The batch it creates is explicitly labelled as a backfill, because that is
   what it is - the link is asserted after the fact, not recorded as it
   happened.

3. **`dvc status` can take a while the first time.** It hashes what it finds,
   and `image_train` holds ~85k files.

4. **Training from the CLI does not refresh the running API.**
   `docker compose exec api python -m src.models.training` runs in a separate
   process, so the API keeps serving the model it loaded at startup and its
   Prometheus counters do not move. Train through `POST /training` instead, or
   restart the container afterwards. This matters for demos.

### Expected layout on disk

```
src/data/raw/
├── X_train_update.csv
├── Y_train_CVw08PX.csv
├── X_test_update.csv
└── image_train/
    └── image_train/
        ├── image_<imageid>_product_<productid>.jpg
        └── ...
```

The official Rakuten dataset extracts with this double nesting
(`image_train/image_train/`). If your structure differs, adjust
`IMAGE_TRAIN_DIR` in `src/config.py`.

## 4. Full startup (first installation)

```bash
# 1. Start services (API + PostgreSQL)
docker compose up -d --build

# 2. Check everything is healthy
docker compose ps
curl http://localhost:8000/health
# → {"status":"ok","database":"up","model_available":false}

# 3. One-shot ingestion of data (text + image path resolution)
docker compose exec api python -m src.data.make_dataset --force

# 4. Train the baseline model (text + image, ResNet18)
#    ⚠️ On the full dataset (~85k rows), image embedding extraction on CPU is long:
#    expect several hours. Keep the terminal running (no sleep mode / SSH drop).
docker compose exec api python -m src.models.training

# 5. Check that the model is loaded by the API
curl http://localhost:8000/health
# → {"status":"ok","database":"up","model_available":true}
curl http://localhost:8000/model
```

### Quick test (subsample)

Pour valider que le pipeline fonctionne bout-en-bout sans attendre l'entraînement
complet, utiliser `--limit` (recommandé : ≥ 3000, sinon certaines classes rares
n'ont pas assez de représentants pour le split stratifié) :

```bash
docker compose exec api python -m src.models.training --limit 3000
```

## 5. Daily usage

```bash
# Start existing services
docker compose up -d

# Stop services (without deleting data)
docker compose down

# Rebuild image after code or requirements.txt changes
docker compose up -d --build

```

> ⚠️ `docker compose build` alone does not recreate the running container —
always use docker compose up -d --build after code changes, otherwise
exec commands will still run on the old container.

## 6. API Endpoints

Documentation interactive : http://localhost:8000/docs

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | API status, database status, model availability |
| GET | `/model` | Metadata and metrics of the currently served model |
| GET | `/metrics` | Prometheus metrics (see `MONITORING_CHECKS.md`) |
| POST | `/training` | (Re)train the model. Optional body: test_size, random_state, limit|
| POST | `/training/trigger` | Body-less retraining trigger, called by the Grafana drift alert |
| POST | `/predict` | Predict prdtypecode for one or multiple observations |
| POST | `/drift/compute` | Recompute data drift and persist it for Grafana |

**Example — quick training:**
```bash
curl -X POST http://localhost:8000/training \
  -H "Content-Type: application/json" \
  -d '{"limit": 3000}'
```

**Example — full training:**
```bash
curl -X POST http://localhost:8000/training \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Example — prédiction :**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "designation": "Retro portable gaming console",
        "description": "400 games included",
        "productid": 123456,
        "imageid": 987654,
        "image_path": null
      }
    ]
  }'
```

> image_path: null is accepted: the image embedding transformer returns
> a zero vector if the image is missing or not found (prediction based on text only).

## 7. Command-line scripts

```bash
# Ingestion (one-shot, idempotent except with --force)
docker compose exec api python -m src.data.make_dataset [--force] [--chunk-size 500]

# Training
docker compose exec api python -m src.models.training [--test-size 0.2] [--random-state 42] [--limit N]

# Prediction on a database sample
docker compose exec api python -m src.models.predict --sample

# Prediction on a JSON file (object or list of objects)
docker compose exec api python -m src.models.predict --input payload.json [--no-log]

# Attach pre-lineage rows to a data batch (local venv, not the container)
python scripts/backfill_data_batches.py
```

## 8. Troubleshooting

**FATAL: password authentication failed or role "..." does not exist"**
- The PostgreSQL volume was initialized with old credentials different from the current .env.
- Postgres only reads POSTGRES_* at volume creation.
-Reset cleanly (⚠️ deletes ingested data, but not the trained model stored in models/`) :
```bash
docker compose down
docker volume ls                              # repérer le volume *_pgdata
docker volume rm <nom_du_volume_pgdata>
docker compose up -d
docker compose exec api python -m src.data.make_dataset --force
```

**The api container does not appear in docker ps**
→ It likely crashed at startup. Check the traceback:
```bash
docker compose ps -a
docker compose logs api --tail=100
```
Common causes: missing dependency in requirements.txt (ModuleNotFoundError), broken import.

**`ValueError: The least populated classes in y have only 1 member`**
- Stratified split fails because --limit is too small.
  Increase --limit (≥ 3000 recommended) or use the full dataset.

**Code changes have no effect after rebuild**
- Ensure the file was saved, then use
docker compose up -d --build (not just docker compose build).

**`dvc pull` returns 403**
- Your DagsHub token is missing or mistyped. Re-run the two
  `dvc remote modify --local` commands from §3.

**`dvc status` says there is nothing tracked**
- Something is hiding the `.dvc` pointers from DVC. Check `git check-ignore -v`
  on one of them: DVC does not read `!` negation patterns in `.gitignore` the
  way git does, which is why the data directories carry their own `.gitignore`
  files instead.

## 9. Objectives and metrics (Phase 1 recap))

* Main metric: weighted F1‑score (imbalanced classes)
* Baseline target: > 0.65
* Obtained result (full dataset, text + image):
    accuracy ≈ 0.822, weighted_f1 ≈ 0.824

## 10. Pipeline architecture

```
DagsHub (DVC remote) — raw files, versioned
        │  dvc pull
        ▼
src/data/raw/  ──►  src/data/make_dataset.py
        │                    │
        │                    ▼
        │        PostgreSQL (raw_samples + data_batches)
        │                    │
        ▼                    ▼
src/features/build_features.py
  ├── TF-IDF (designation) ──┐
  ├── TF-IDF (description) ──┤
  ├── Scaler (productid, imageid) ──┤──► ColumnTransformer ──► LogisticRegression
  └── ImageEmbeddingTransformer (image_path, ResNet18 frozen ) ──┘
        │
        ▼
MLflow Model Registry (Production / Staging)
        │
        ▼
src/api/main.py — /training, /training/trigger, /predict, /health, /model,
                  /metrics, /drift/compute
        │
        ▼
Prometheus ──► Grafana (dashboards + alerts ──► retraining webhook)
```