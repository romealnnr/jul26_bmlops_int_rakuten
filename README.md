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

## 1. Prérequis

- Docker et Docker Compose
- Les fichiers de données brutes du challenge Rakuten (voir §3)

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
```

> ⚠️ **Important** : `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` ne sont
> lus par Postgres **qu'à la toute première initialisation** du volume de
> données. Si tu changes ces valeurs après coup, Postgres les ignorera et
> gardera les anciens identifiants → erreurs d'authentification. Dans ce cas,
> supprimer le volume (§7 Dépannage) pour forcer une réinitialisation propre.

## 3. Données brutes attendues

Placer dans `src/data/raw/` :

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

> Le dataset officiel Rakuten France Multimodal Product Classification
> s'extrait avec cette double imbrication (`image_train/image_train/`).
> Si ta structure diffère, ajuste `IMAGE_TRAIN_DIR` dans `src/config.py`.

## 4. Démarrage complet (première installation)

```bash
# 1. Démarrer les services (API + PostgreSQL)
docker compose up -d --build

# 2. Vérifier que tout est sain
docker compose ps
curl http://localhost:8000/health
# → {"status":"ok","database":"up","model_available":false}

# 3. Ingestion one-shot des données (texte + résolution des chemins d'images)
docker compose exec api python -m src.data.make_dataset --force

# 4. Entraînement du modèle baseline (texte + image, ResNet18)
#    ⚠️ Sur le dataset complet (~85 000 lignes), l'extraction d'embeddings
#    image en CPU est longue : compter plusieurs heures. Prévoir de laisser
#    tourner sans interrompre le terminal (pas de veille PC / coupure SSH).
docker compose exec api python -m src.models.training

# 5. Vérifier que le modèle est bien chargé par l'API
curl http://localhost:8000/health
# → {"status":"ok","database":"up","model_available":true}
curl http://localhost:8000/model
```

### Test rapide (sous-échantillon)

Pour valider que le pipeline fonctionne bout-en-bout sans attendre l'entraînement
complet, utiliser `--limit` (recommandé : ≥ 3000, sinon certaines classes rares
n'ont pas assez de représentants pour le split stratifié) :

```bash
docker compose exec api python -m src.models.training --limit 3000
```

## 5. Utilisation quotidienne

```bash
# Démarrer les services existants
docker compose up -d

# Arrêter les services (sans supprimer les données)
docker compose down

# Reconstruire l'image après une modification de code ou requirements.txt
docker compose up -d --build
```

> ⚠️ `docker compose build` seul ne recrée **pas** le conteneur en cours
> d'exécution — toujours utiliser `docker compose up -d --build` après une
> modification de code, sinon les commandes `exec` continueront de tourner
> sur l'ancienne image/conteneur.

## 6. Endpoints API

Documentation interactive : http://localhost:8000/docs

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | État de l'API, de la base et disponibilité du modèle |
| GET | `/model` | Métadonnées et métriques du modèle actuellement servi |
| POST | `/training` | (Ré)entraîne le modèle. Body optionnel : `test_size`, `random_state`, `limit` |
| POST | `/predict` | Prédit `prdtypecode` pour une ou plusieurs observations |

**Exemple — entraînement (test rapide) :**
```bash
curl -X POST http://localhost:8000/training \
  -H "Content-Type: application/json" \
  -d '{"limit": 3000}'
```

**Exemple — entraînement complet :**
```bash
curl -X POST http://localhost:8000/training \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Exemple — prédiction :**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "designation": "Console de jeu portable retro",
        "description": "400 jeux inclus",
        "productid": 123456,
        "imageid": 987654,
        "image_path": null
      }
    ]
  }'
```

> `image_path: null` est accepté : le transformer d'embeddings image renvoie
> un vecteur nul si l'image est absente ou introuvable (prédiction basée sur
> le texte seul dans ce cas, pas d'erreur).

## 7. Scripts en ligne de commande

```bash
# Ingestion (one-shot, idempotent sauf --force)
docker compose exec api python -m src.data.make_dataset [--force] [--chunk-size 500]

# Entraînement
docker compose exec api python -m src.models.training [--test-size 0.2] [--random-state 42] [--limit N]

# Prédiction sur un échantillon de la base
docker compose exec api python -m src.models.predict --sample

# Prédiction sur un fichier JSON (objet ou liste d'objets)
docker compose exec api python -m src.models.predict --input payload.json [--no-log]
```

## 8. Dépannage

**`FATAL: password authentication failed`** ou **`role "..." does not exist`**
→ Le volume PostgreSQL a été initialisé avec d'anciens identifiants différents
de `.env` actuel (Postgres ne relit `POSTGRES_*` qu'à la création du volume).
Réinitialiser proprement (⚠️ supprime les données ingérées, pas le modèle
entraîné qui est sur disque dans `models/`) :
```bash
docker compose down
docker volume ls                              # repérer le volume *_pgdata
docker volume rm <nom_du_volume_pgdata>
docker compose up -d
docker compose exec api python -m src.data.make_dataset --force
```

**Le conteneur `api` n'apparaît pas dans `docker ps`**
→ Il a probablement crashé au démarrage. Voir le traceback :
```bash
docker compose ps -a
docker compose logs api --tail=100
```
Causes fréquentes : dépendance manquante dans `requirements.txt`
(`ModuleNotFoundError`), import cassé entre fichiers après une modification.

**`ValueError: The least populated classes in y have only 1 member`**
→ Le split stratifié échoue car `--limit` est trop petit : certaines classes
rares n'ont qu'un seul représentant dans l'échantillon. Augmenter `--limit`
(≥ 3000 recommandé) ou ne pas le spécifier pour utiliser le dataset complet.

**Une modification de code n'a aucun effet après rebuild**
→ Vérifier que le fichier a bien été sauvegardé localement, puis utiliser
`docker compose up -d --build` (pas seulement `docker compose build`) pour
recréer le conteneur avec la nouvelle image.

## 9. Objectifs et métriques (rappel Phase 1)

- **Métrique principale** : weighted F1-score (classes déséquilibrées)
- **Cible baseline** : > 0.65
- **Résultat obtenu** (dataset complet, texte + image) : `accuracy ≈ 0.822`,
  `weighted_f1 ≈ 0.824`

## 10. Architecture du pipeline

```
PostgreSQL (raw_samples, features en JSONB)
        │
        ▼
src/features/build_features.py
  ├── TF-IDF (designation) ──┐
  ├── TF-IDF (description) ──┤
  ├── Scaler (productid, imageid) ──┤──► ColumnTransformer ──► LogisticRegression
  └── ImageEmbeddingTransformer (image_path, ResNet18 gelé) ──┘
        │
        ▼
models/baseline_logreg.joblib (pipeline complet sauvegardé)
        │
        ▼
src/api/main.py — endpoints /training, /predict, /health, /model
```