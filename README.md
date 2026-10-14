Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

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
# Key for /training, /training/trigger and /drift/compute (X-API-Key header)
API_KEY=<python -c "import secrets; print(secrets.token_urlsafe(32))">
# Unlocks the Retrain / Compute drift forms in Streamlit
STREAMLIT_ADMIN_PASSWORD=<choose a password>

# --- MLflow ---
MLFLOW_TRACKING_URI=postgresql+psycopg://"POSTGRES_USER":"POSTGRES_PASSWORD"@localhost:5432/mlops
# replace "POSTGRES_PASSWORD" and "POSTGRES_PASSWORD"
MLFLOW_EXPERIMENT_NAME=rakuten-baseline-v2
MLFLOW_REGISTERED_MODEL_NAME=rakuten_baseline

# --- Airflow ---
AIRFLOW_UID=1000
AIRFLOW_FERNET_KEY=<create a key>
AIRFLOW_JWT_SECRET=<create a key>
# Airflow web login (http://localhost:8081)
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=<choose a password>

# --- Grafana ---
# Required: docker compose refuses to start without it (no admin/admin fallback).
GRAFANA_ADMIN_PASSWORD=<choose a password>
```

> Postgres (`POSTGRES_PORT`) and Grafana (port 3000) are published on 127.0.0.1
only, so they are reachable from this machine but not from the network. To open
Grafana to the network on purpose, set `GRAFANA_BIND_ADDRESS=0.0.0.0` in `.env`.
Grafana reads `GRAFANA_ADMIN_PASSWORD` only when its data volume is first created;
on an existing volume set it with
`docker compose exec grafana grafana cli admin reset-admin-password "<new password>"`.

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
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"limit": 3000}'
```

**Example — full training:**
```bash
curl -X POST http://localhost:8000/training \
  -H "X-API-Key: $API_KEY" \
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

## 9. Objectives and metrics (Phase 1 recap))

* Main metric: weighted F1‑score (imbalanced classes)
* Baseline target: > 0.65
* Obtained result (full dataset, text + image):
    accuracy ≈ 0.822, weighted_f1 ≈ 0.824

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
