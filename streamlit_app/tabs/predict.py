import hmac
import json
import random
from pathlib import Path

import pandas as pd
import requests
import streamlit as stlt
from records import records_for_api

from config import (
    API_IMAGE_ROOT,
    API_INTERNAL_URL,
    API_KEY,
    PRDTYPECODE_LABELS,
    RAW_DIR,
    SPLITS,
    STREAMLIT_ADMIN_PASSWORD,
    Y_TRAIN_FILE,
)

EXAMPLE_RECORD = {
    "designation": "Console de jeu portable retro",
    "description": "400 jeux inclus, écran couleur",
    "productid": 123456,
    "imageid": 987654,
    "image_path": None,
}

# Fields /predict accepts per record (src/api/schemas.py ProductRecord).
API_RECORD_FIELDS = [
    "designation", "description", "productid", "imageid", "image_path", "true_label",
]


def label(code) -> str:
    try:
        return f"{int(code)} ({PRDTYPECODE_LABELS.get(int(code), 'unknown')})"
    except (TypeError, ValueError):
        return str(code)
    

def to_df(payload: dict) -> pd.DataFrame:
    """Prediction results as a table. The flag comes from the API, never recomputed here."""
    return pd.DataFrame(payload["results"])


def call_api(method: str, path: str, timeout: int = 30, admin: bool = False, **kwargs):
    """Call the API and return (ok, payload). Never raises.

    admin=True sends API_KEY as X-API-Key (needed by /training and /drift/compute).
    """
    if admin:
        kwargs["headers"] = {**kwargs.get("headers", {}), "X-API-Key": API_KEY}
    try:
        r = requests.request(method, f"{API_INTERNAL_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        return False, f"Connection error: {exc}"
    
    try:
        data = r.json()
    except ValueError:
        data = r.text
    
    if r.ok:
        return True, data
    
    detail = data.get("detail", data) if isinstance(data, dict) else data
    
    return False, f"HTTP {r.status_code}: {detail}"


def admin_unlocked() -> bool:
    """True when STREAMLIT_ADMIN_PASSWORD is set and the visitor typed it (constant-time)."""
    if not STREAMLIT_ADMIN_PASSWORD:
        stlt.info("Admin actions are disabled: set STREAMLIT_ADMIN_PASSWORD in .env to enable.")
        return False
    entered = stlt.text_input("Admin password", type="password", key="admin_password")
    if not entered:
        return False
    if not hmac.compare_digest(entered.encode(), STREAMLIT_ADMIN_PASSWORD.encode()):
        stlt.error("Wrong admin password.")
        return False
    return True


def show_predictions(payload: dict):
    df = to_df(payload)
    if "flagged" not in df.columns:
        stlt.warning("The API response has no 'flagged' field (older API version).")
        df["flagged"] = False
    n_flagged = int(df["flagged"].sum())
    threshold = payload.get("flag_threshold")
    msg = f"{payload['count']} prediction(s), {n_flagged} flagged by the API"
    if threshold is not None:
        msg += f" (confidence < {float(threshold):.0%})"
    stlt.success(msg)
    stlt.dataframe(df, use_container_width=True)
    if len(df) == 1:
        row = df.iloc[0]
        c1, c2, c3 = stlt.columns(3)
        c1.metric("Predicted class", label(row["prediction"]))
        c2.metric("Probability", f"{row['probability']:.1%}")
        c3.metric("Model version", row["model_version"])
        if row["flagged"]:
            stlt.warning(f"Flagged by the API: {row.get('flag_reason') or 'no reason given'}")


def show_model_info(info: dict):
    c1, c2, c3 = stlt.columns(3)
    c1.metric("Model", info["model_name"])
    c2.metric("Version", info["version"])
    c3.metric("Features", info["n_features"])
    stlt.caption(f"Trained at: {info['trained_at']}")

    metrics = info.get("metrics") or {}
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    other = {k: v for k, v in metrics.items() if k not in numeric}
    if numeric:
        cols = stlt.columns(min(len(numeric), 4))
        for i, (k, v) in enumerate(numeric.items()):
            cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v, float) else v)
    if other:
        stlt.json(other)


def available_splits() -> list[str]:
    """Only the splits whose X_<split>_update.csv is actually present in RAW_DIR."""
    return [s for s in SPLITS if (RAW_DIR / f"X_{s}_update.csv").is_file()]


@stlt.cache_data
def load_split(split: str) -> pd.DataFrame:
    x = pd.read_csv(RAW_DIR / f"X_{split}_update.csv", index_col=0)
    x["designation"] = x["designation"].fillna("")
    x["description"] = x["description"].fillna("")
    if split == "train":
        y = pd.read_csv(RAW_DIR / Y_TRAIN_FILE, index_col=0)
        x = x.join(y)  # adds the target column (e.g. prdtypecode)
    return x


def image_location(split: str, imageid: int, productid: int):
    """Return (local_path, api_path) for a product photo, or (None, None) if absent.

    The Rakuten archive unpacks into a nested folder (image_train/image_train/<file>);
    a flat layout (image_train/<file>) is also accepted. The API path mirrors the
    folder where the file was actually found, because the API silently falls back to
    a text-only prediction when it cannot open image_path.
    """
    fname = f"image_{imageid}_product_{productid}.jpg"
    nested = Path(f"image_{split}") / f"image_{split}" / fname
    flat = Path(f"image_{split}") / fname
    for rel in (nested, flat):
        if (RAW_DIR / rel).is_file():
            return RAW_DIR / rel, f"{API_IMAGE_ROOT}/{rel.as_posix()}"
    return None, None


def _random_row(n: int):
    stlt.session_state["row_pos"] = random.randrange(n)


def dataset_sample_mode():
    splits = available_splits()
    missing = [s for s in SPLITS if s not in splits]
    if not splits:
        stlt.error(
            f"No dataset CSV found in {RAW_DIR} (expected X_train_update.csv). "
            "Is ./src/data/raw mounted into the streamlit container?"
        )
        return
    split = stlt.radio("Split", splits, horizontal=True)
    if missing:
        stlt.caption(
            "Not available: "
            + ", ".join(f"'{s}' (X_{s}_update.csv is not in {RAW_DIR})" for s in missing)
            + ". Only splits present in the data folder are offered."
        )

    try:
        df = load_split(split)
    except (FileNotFoundError, pd.errors.ParserError, KeyError) as exc:
        stlt.error(f"Cannot read the {split} split from {RAW_DIR}: {exc}")
        return

    stlt.session_state.setdefault("row_pos", 0)
    if stlt.session_state["row_pos"] >= len(df):
        stlt.session_state["row_pos"] = 0

    c1, c2 = stlt.columns([1, 3])
    c1.button("🎲 Random row", on_click=_random_row, args=(len(df),))
    pos = c2.number_input("Row position", 0, len(df) - 1, key="row_pos")

    row = df.iloc[int(pos)]
    fname = f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"
    local_img, api_img = image_location(split, int(row["imageid"]), int(row["productid"]))

    left, right = stlt.columns([1, 2])
    with left:
        if local_img is not None:
            stlt.image(str(local_img), use_container_width=True)
        else:
            stlt.warning(
                f"Image not found under {RAW_DIR}/image_{split}/ (nested or flat): {fname}. "
                "The prediction will be text-only."
            )
    with right:
        stlt.markdown(f"**{row['designation']}**")
        stlt.caption(row["description"][:500] or "(no description)")
        stlt.write(f"productid: `{int(row['productid'])}`, imageid: `{int(row['imageid'])}`")
        target_col = df.columns[-1] if split == "train" else None
        if target_col:
            stlt.write(f"True label: **{label(row[target_col])}**")

    use_image = stlt.checkbox("Send image_path to the API", value=True,
                            help="Untick to see how the prediction changes without the image.")
    if use_image and api_img:
        stlt.caption(f"image_path sent to the API: `{api_img}`")
    elif use_image:
        stlt.caption("No image file found: image_path is sent as null (text-only prediction).")

    if stlt.button("Predict this product"):
        record = {
            "designation": row["designation"],
            "description": row["description"],
            "productid": int(row["productid"]),
            "imageid": int(row["imageid"]),
            "image_path": api_img if use_image else None,
        }
        ok, data = call_api("POST", "/predict", json={"records": [record]})
        if ok:
            show_predictions(data)
            if target_col:
                pred = data["results"][0]["prediction"]
                stlt.info(f"Predicted: {label(pred)}, true label: {label(row[target_col])}")
        else:
            stlt.error(data)




def render():
    # status
    stlt.subheader("API & model status")
    col1, col2 = stlt.columns(2)
    with col1:
        if stlt.button("Check health"):
            ok, data = call_api("GET", "/health", timeout=5)
            if ok:
                stlt.json(data)
            else:
                stlt.error(data)
    with col2:
        if stlt.button("Show model info"):
            ok, data = call_api("GET", "/model", timeout=10)
            show_model_info(data) if ok else stlt.error(data)

    stlt.divider()

    # predict
    stlt.subheader("Prediction")
    mode = stlt.radio(
        "Input mode", ["Dataset sample", "Single product", "CSV batch", "Raw JSON"], horizontal=True
    )

    if mode == "Dataset sample":
        dataset_sample_mode()
    elif mode == "Single product":
        with stlt.form("predict_single"):
            designation = stlt.text_input("designation", EXAMPLE_RECORD["designation"])
            description = stlt.text_area("description", EXAMPLE_RECORD["description"])
            c1, c2 = stlt.columns(2)
            productid = c1.text_input("productid (optional)", "")
            imageid = c2.text_input("imageid (optional)", "")
            image_path = stlt.text_input(
                "image_path (optional)",
                "",
                help=("Path as seen INSIDE the api container. "
                      "Empty or unreadable = no image information."
                      ),
            )
            submitted = stlt.form_submit_button("Predict")
        if submitted:
            try:
                record = {
                    "designation": designation,
                    "description": description,
                    "productid": int(productid) if productid.strip() else None,
                    "imageid": int(imageid) if imageid.strip() else None,
                    "image_path": image_path.strip() or None,
                }
            except ValueError:
                stlt.error("productid and imageid must be whole numbers (or empty).")
            else:
                ok, data = call_api("POST", "/predict", json={"records": [record]})
                show_predictions(data) if ok else stlt.error(data)

    elif mode == "CSV batch":
        stlt.caption(
            "CSV columns: designation, description, productid, imageid, image_path, "
            "true_label (optional)"
        )
        file = stlt.file_uploader("CSV file", type="csv")
        if file is not None:
            df = pd.read_csv(file)
            stlt.dataframe(df.head(), use_container_width=True)
            max_rows = stlt.number_input("Max rows to send", 1, 1000, min(len(df), 100))
            if stlt.button("Predict batch"):
                sample = df.head(int(max_rows)).astype(object)
                # /predict rejects unknown fields (422): send only the accepted columns.
                dropped = [c for c in sample.columns if c not in API_RECORD_FIELDS]
                if dropped:
                    stlt.caption(f"Not sent to the API: {', '.join(map(str, dropped))}")
                records = records_for_api(sample, API_RECORD_FIELDS)
                ok, data = call_api("POST", "/predict", json={"records": records}, timeout=120)
                if ok:
                    show_predictions(data)
                    out = pd.concat(
                        [sample.reset_index(drop=True), to_df(data)],
                        axis=1,
                    )
                    stlt.download_button(
                        "Download results (CSV)",
                        out.to_csv(index=False).encode(),
                        "predictions.csv",
                        "text/csv",
                    )
                else:
                    stlt.error(data)

    else:  # Raw JSON
        raw = stlt.text_area(
            "Request body",
            json.dumps({"records": [EXAMPLE_RECORD]}, indent=2, ensure_ascii=False),
            height=260,
        )
        if stlt.button("Send"):
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                stlt.error(f"Invalid JSON: {exc}")
            else:
                ok, data = call_api("POST", "/predict", json=body)
                show_predictions(data) if ok else stlt.error(data)

    stlt.divider()

    # admin
    stlt.subheader("Admin actions")
    stlt.caption(
        "These change state (model files, MLflow runs, DB). "
        "Use carefully during demonstration."
    )
    if not admin_unlocked():
        return

    with stlt.expander("Retrain model (/training)"):
        with stlt.form("train"):
            c1, c2, c3 = stlt.columns(3)
            test_size = c1.number_input("test_size", 0.05, 0.95, 0.2, 0.05)
            random_state = c2.number_input("random_state", 0, 10_000, 42)
            limit = c3.number_input("limit (0 = all rows)", 0, 1_000_000, 1000)
            continuous = stlt.checkbox("continuous (next slice from pipeline_cursor)")
            go = stlt.form_submit_button("Start training")
        if go:
            body = {
                "test_size": float(test_size),
                "random_state": int(random_state),
                "limit": int(limit) or None,
                "continuous": continuous,
            }
            with stlt.spinner("Training... this can take a while"):
                ok, data = call_api("POST", "/training", json=body, timeout=1800, admin=True)
            if ok and data.get("status") == "no_new_data":
                stlt.info(data.get("detail") or "No new rows: nothing was trained.")
                stlt.json(data)
            elif ok:
                decision = (data.get("comparison") or {}).get("decision", "")
                stlt.success(f"Trained registry version {data['version']} ({decision})")
                stlt.json(data)
            else:
                stlt.error(data)

    with stlt.expander("Compute data drift (/drift/compute)"):
        if stlt.button("Compute drift"):
            with stlt.spinner("Computing drift..."):
                ok, data = call_api("POST", "/drift/compute", timeout=300, admin=True)
            if ok:
                stlt.json(data)
                stlt.info("Metrics are stored in the DB, check the Grafana tab.")
            else:
                stlt.error(data)
