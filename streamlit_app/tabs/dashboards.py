import streamlit as stlt
import streamlit.components.v1 as components

from config import DASHBOARDS


def render():
    name = stlt.radio("Dashboard", list(DASHBOARDS), horizontal=True)
    cfg = DASHBOARDS[name]

    if not cfg["url"]:
        stlt.warning(f"No URL configured for {name}.")
        return

    stlt.link_button(f"Open {name} in a new tab", cfg["url"])
    if cfg.get("help"):
        stlt.caption(cfg["help"])

    if cfg["embed"]:
        # The browser loads this URL directly, so it must be reachable from the
        # user's machine (localhost ports), not a Docker service name.
        components.iframe(cfg["url"], height=800, scrolling=True)
    else:
        stlt.info(f"{name} can't be embedded, use the button above.")