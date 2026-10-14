# jul26_bmlops_int_rakuten: Rakuten product classification (MLOps)

Multiclass classification of Rakuten products (`prdtypecode`, 27 classes) from the
product text (`designation`, `description`) and the product image.
Multimodal baseline: TF-IDF (text) and frozen ResNet18 embeddings (image), combined in
a scikit-learn pipeline with a logistic-regression classifier. The stack around it:
PostgreSQL, FastAPI, MLflow tracking and registry, Airflow continuous training,
Evidently drift, Prometheus, Grafana and a Streamlit front end, all in Docker Compose.

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

## 3. Expected raw data

Place the following in `src/data/raw/` :

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

> The official Rakuten France Multimodal Product Classification dataset
> extracts with this double nesting (image_train/image_train/).
> If your structure differs, adjust IMAGE_TRAIN_DIR in src/config.py.

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
| POST | `/training` | (Re)train the model. Optional body: test_size, random_state, limit|
| POST | `/predict` | Predict prdtypecode for one or multiple observations |

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

## 9. Objectives and metrics

* Main metric: weighted F1-score (imbalanced classes)
* Baseline target: > 0.65
* Score of the model the API serves right now: read it from the stack, not from this file.
  `curl http://localhost:8000/model` returns the `metrics` block (accuracy, weighted_f1,
  weighted_precision, weighted_recall) of the served model, and the same values are logged
  on its MLflow run.

> Earlier versions of this README quoted accuracy ≈ 0.822 and weighted_f1 ≈ 0.824
> ("full dataset, text + image"). That figure was removed because it does not describe
> the model this code serves:
> - it came from one training on the full dataset, while continuous training (the
>   Airflow DAG `rakuten_continuous_training`, or `python -m src.models.training --continuous`)
>   trains each model on batches of `CONTINUOUS_BATCH_SIZE = 3000` new rows
>   (`src/models/training.py`), so the served model's score is not comparable to it;
> - it was measured with `productid` and `imageid` among the model inputs. These
>   catalogue identifiers can carry the category on their own, so a score from a
>   model that uses them may overstate how well it classifies new products.

## 10. Pipeline architecture

```
PostgreSQL (raw_samples, features en JSONB)
        │
        ▼
src/features/build_features.py
  ├── TF-IDF (designation) ──┐
  ├── TF-IDF (description) ──┤
  ├── Scaler (productid, imageid) ──┤──► ColumnTransformer ──► LogisticRegression
  └── ImageEmbeddingTransformer (image_path, ResNet18 frozen ) ──┘
        │
        ▼
models/baseline_logreg.joblib (full pipeline saved)
        │
        ▼
src/api/main.py — endpoints /training, /predict, /health, /model
```
