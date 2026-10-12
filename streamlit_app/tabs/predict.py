import json

import pandas as pd
import requests
import streamlit as stlt

from config import API_INTERNAL_URL, FLAG_THRESHOLD


EXAMPLE_RECORD = {
    "designation": "Console de jeu portable retro",
    "description": "400 jeux inclus, écran couleur",
    "productid": 123456,
    "imageid": 987654,
    "image_path": None,
}

def to_df(payload: dict) -> pd.DataFrame:
    """ Adds a results flag column to the prediction results """
    df = pd.DataFrame(payload["results"])
    df["flagged"] = df["probability"] < FLAG_THRESHOLD
    return df


def call_api(method: str, path: str, timeout: int = 30, **kwargs):
    """Call the API and return (ok, payload). Never raises."""
    
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


def show_predictions(payload: dict):
    df = to_df(payload)
    n_flagged = int(df["flagged"].sum())
    stlt.success(
        f"{payload['count']} prediction(s), {n_flagged} flagged "
        f"(confidence < {FLAG_THRESHOLD:.0%})"
    )
    stlt.dataframe(df, use_container_width=True)
    if len(df) == 1:
        row = df.iloc[0]
        c1, c2, c3 = stlt.columns(3)
        c1.metric("Predicted class", int(row["prediction"]))
        c2.metric("Probability", f"{row['probability']:.1%}")
        c3.metric("Model version", row["model_version"])
        if row["flagged"]:
            stlt.warning("Low confidence: this prediction would be flagged.")


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
        "Input mode", ["Single product", "CSV batch", "Raw JSON"], horizontal=True
    )

    if mode == "Single product":
        with stlt.form("predict_single"):
            designation = stlt.text_input("designation", EXAMPLE_RECORD["designation"])
            description = stlt.text_area("description", EXAMPLE_RECORD["description"])
            c1, c2 = stlt.columns(2)
            productid = c1.text_input("productid (optional)", "")
            imageid = c2.text_input("imageid (optional)", "")
            image_path = stlt.text_input(
                "image_path (optional)",
                "",
                help="Path as seen INSIDE the api container. Empty or unreadable = no image information.",
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
            "CSV columns: designation, description, productid, imageid, image_path"
        )
        file = stlt.file_uploader("CSV file", type="csv")
        if file is not None:
            df = pd.read_csv(file)
            stlt.dataframe(df.head(), use_container_width=True)
            max_rows = stlt.number_input("Max rows to send", 1, 1000, min(len(df), 100))
            if stlt.button("Predict batch"):
                sample = df.head(int(max_rows)).astype(object)
                records = sample.where(sample.notna(), None).to_dict("records")
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
    stlt.caption("These change state (model files, MLflow runs, DB). Use carefully during demonstration.")

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
                ok, data = call_api("POST", "/training", json=body, timeout=1800)
            if ok:
                stlt.success(f"Trained version {data['version']}")
                stlt.json(data)
            else:
                stlt.error(data)

    with stlt.expander("Compute data drift (/drift/compute)"):
        if stlt.button("Compute drift"):
            with stlt.spinner("Computing drift..."):
                ok, data = call_api("POST", "/drift/compute", timeout=300)
            if ok:
                stlt.json(data)
                stlt.info("Metrics are stored in the DB, check the Grafana tab.")
            else:
                stlt.error(data)
