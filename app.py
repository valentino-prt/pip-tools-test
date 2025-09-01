import os
import sys
import time
import io
import hashlib
import signal
import threading
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Session context (Streamlit) ----
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    get_script_run_ctx = None

# =========================
#        SETTINGS
# =========================
SHUTDOWN_TIMEOUT = 15   # seconds with no sessions before exit
HEARTBEAT_INTERVAL = 5  # seconds
DEFAULT_PORT = 8501

st.set_page_config(page_title="Streamlit Auto-shutdown + Histogram Editor", layout="wide")

# =========================
#  AUTO-SHUTDOWN MECHANISM
# =========================
class SessionRegistry:
    def __init__(self):
        self.sessions: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.monitor_started = False

    def heartbeat(self, sid: str):
        with self.lock:
            self.sessions[sid] = time.time()

    def prune_and_count(self, stale_after: int):
        now = time.time()
        with self.lock:
            self.sessions = {sid: ts for sid, ts in self.sessions.items() if now - ts <= stale_after}
            return len(self.sessions)

def _hard_exit():
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)
    except Exception:
        pass
    os._exit(0)

def start_monitor(registry: SessionRegistry, timeout: int):
    def loop():
        empty_count = 0
        while True:
            alive = registry.prune_and_count(timeout)
            if alive == 0:
                empty_count += 1
            else:
                empty_count = 0
            if empty_count >= 3:  # ~ 3 * 5s
                print("🛑 No active sessions → shutting down.")
                _hard_exit()
            time.sleep(5)
    threading.Thread(target=loop, daemon=True).start()

def get_session_id():
    if get_script_run_ctx:
        ctx = get_script_run_ctx()
        if ctx and getattr(ctx, "session_id", None):
            return ctx.session_id
    import uuid
    return str(uuid.uuid4())

@st.cache_resource
def get_registry():
    return SessionRegistry()

registry = get_registry()
if not registry.monitor_started:
    start_monitor(registry, SHUTDOWN_TIMEOUT)
    registry.monitor_started = True

sid = get_session_id()
registry.heartbeat(sid)

# keep heartbeating while tab is open
if hasattr(st, "autorefresh"):
    st.autorefresh(interval=HEARTBEAT_INTERVAL * 1000, key="__hb__")

# =========================
#    FILE I/O UTILITIES
# =========================
def parse_lines_to_df(text: str) -> pd.DataFrame:
    """
    Parse a multi-line string where each line is a histogram (values separated by commas or spaces).
    Returns a rectangular DataFrame (pads shorter rows with NaN).
    """
    rows: List[List[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # split on comma/space
        parts = [p for p in line.replace(",", " ").split(" ") if p != ""]
        vals: List[float] = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                # ignore non-numeric tokens gracefully
                pass
        if vals:
            rows.append(vals)
    if not rows:
        return pd.DataFrame()

    max_len = max(len(r) for r in rows)
    padded = [r + [np.nan] * (max_len - len(r)) for r in rows]
    df = pd.DataFrame(padded, index=[f"hist_{i+1}" for i in range(len(padded))],
                      columns=[f"bin_{j+1}" for j in range(max_len)])
    return df

def df_to_lines(df: pd.DataFrame) -> str:
    """
    Convert DF back to lines (comma-separated). NaN are dropped from line end.
    """
    lines = []
    for _, row in df.iterrows():
        vals = row.values.tolist()
        # trim trailing NaNs
        while vals and (vals[-1] is None or (isinstance(vals[-1], float) and np.isnan(vals[-1]))):
            vals.pop()
        line = ",".join(str(v) for v in vals)
        lines.append(line)
    return "\n".join(lines)

def file_hash_and_mtime(path: str) -> Tuple[str, float]:
    with open(path, "rb") as f:
        data = f.read()
    h = hashlib.sha256(data).hexdigest()
    mtime = os.path.getmtime(path)
    return h, mtime

def load_from_path(path: str) -> Tuple[pd.DataFrame, str, float]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    df = parse_lines_to_df(text)
    h, mtime = file_hash_and_mtime(path)
    return df, h, mtime

def load_from_bytes(name: str, data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8", errors="ignore")
    return parse_lines_to_df(text)

def safe_write_to_path(path: str, df: pd.DataFrame) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(df_to_lines(df))
    os.replace(tmp, path)

# =========================
#      DIFF / MERGE
# =========================
def diff_cells(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Return boolean DF (True if different). Aligns indexes/columns."""
    a2, b2 = a.align(b, join="outer")
    return ~(a2.eq(b2) | (a2.isna() & b2.isna()))

def compute_conflicts(original: pd.DataFrame, ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """
    3-way merge conflicts mask (True where both ours and theirs changed from original and differ).
    """
    o, r_ours = original.align(ours, join="outer")
    o2, r_theirs = original.align(theirs, join="outer")
    changed_ours = diff_cells(o, r_ours)
    changed_theirs = diff_cells(o2, r_theirs)
    both_changed = changed_ours & changed_theirs
    different_values = diff_cells(r_ours, r_theirs)
    return both_changed & different_values

def merge_3way(original: pd.DataFrame, ours: pd.DataFrame, theirs: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    strategy: 'ours', 'theirs', or 'overwrite' (ours wins everywhere)
    """
    o, r_ours = original.align(ours, join="outer")
    o2, r_theirs = original.align(theirs, join="outer")
    # start from original
    merged = o.copy()

    # cells changed in ours
    ours_mask = diff_cells(o, r_ours)
    # cells changed in theirs
    theirs_mask = diff_cells(o2, r_theirs)

    # apply non-conflicting changes
    only_ours = ours_mask & ~theirs_mask
    only_theirs = theirs_mask & ~ours_mask
    merged[only_ours] = r_ours[only_ours]
    merged[only_theirs] = r_theirs[only_theirs]

    # conflicts
    conflicts = ours_mask & theirs_mask & diff_cells(r_ours, r_theirs)
    if strategy == "ours" or strategy == "overwrite":
        merged[conflicts] = r_ours[conflicts]
    elif strategy == "theirs":
        merged[conflicts] = r_theirs[conflicts]
    else:
        # default to ours
        merged[conflicts] = r_ours[conflicts]

    return merged

# =========================
#        UI / STATE
# =========================
if "df_original" not in st.session_state:
    st.session_state.df_original: Optional[pd.DataFrame] = None
if "df_current" not in st.session_state:
    st.session_state.df_current: Optional[pd.DataFrame] = None
if "source_kind" not in st.session_state:
    st.session_state.source_kind = None  # "path" | "upload"
if "source_path" not in st.session_state:
    st.session_state.source_path = ""
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "file_mtime" not in st.session_state:
    st.session_state.file_mtime = None

st.title("📊 Histogram Editor + Comparator + Auto-shutdown")

with st.expander("ℹ️ Format du fichier attendu", expanded=False):
    st.write(
        "- Chaque **ligne** est un histogramme.\n"
        "- Les **valeurs** d’un histogramme sont séparées par **virgules** ou **espaces**.\n"
        "- Exemple :\n"
        "```\n"
        "1,2,3,2,1\n"
        "0 1 4 1 0\n"
        "2,2,2,2\n"
        "```"
    )

col_src_left, col_src_right = st.columns([2, 1])

with col_src_left:
    src_tab = st.tabs(["📁 Chemin local", "📤 Upload fichier"])
    with src_tab[0]:
        path = st.text_input("Chemin du fichier (lecture/écriture)", value=st.session_state.get("source_path", ""))
        load_btn_path = st.button("Load depuis le chemin", type="primary", use_container_width=True)
        if load_btn_path:
            try:
                df, h, m = load_from_path(path)
                st.session_state.df_original = df.copy()
                st.session_state.df_current = df.copy()
                st.session_state.source_kind = "path"
                st.session_state.source_path = path
                st.session_state.file_hash = h
                st.session_state.file_mtime = m
                st.success(f"Chargé depuis {path} — {df.shape[0]} histogrammes, {df.shape[1]} bins.")
            except Exception as e:
                st.error(f"Erreur chargement: {e}")

    with src_tab[1]:
        up = st.file_uploader("Upload (lecture seule, téléchargement pour sauvegarder)", type=["txt", "csv", "dat"])
        load_btn_upload = st.button("Load depuis l'upload", use_container_width=True)
        if load_btn_upload:
            if up is None:
                st.warning("Sélectionne d’abord un fichier.")
            else:
                try:
                    data = up.read()
                    df = load_from_bytes(up.name, data)
                    st.session_state.df_original = df.copy()
                    st.session_state.df_current = df.copy()
                    st.session_state.source_kind = "upload"
                    st.session_state.source_path = up.name
                    st.session_state.file_hash = None
                    st.session_state.file_mtime = None
                    st.success(f"Chargé depuis upload — {df.shape[0]} histogrammes, {df.shape[1]} bins.")
                except Exception as e:
                    st.error(f"Erreur chargement: {e}")

with col_src_right:
    if st.session_state.df_current is not None:
        st.metric("Histos", st.session_state.df_current.shape[0])
        st.metric("Bins max", st.session_state.df_current.shape[1])

# =========================
#       EDITOR / PLOTS
# =========================
if st.session_state.df_current is not None:
    st.subheader("📝 Édition des données")
    st.caption("Double-clique pour éditer les cellules. Tu peux ajouter/supprimer des lignes/colonnes via les options de l’éditeur.")
    edited_df = st.data_editor(
        st.session_state.df_current,
        num_rows="dynamic",
        use_container_width=True,
        key="__editor__",
    )
    # Sync back
    st.session_state.df_current = edited_df

    st.subheader("📈 Visualisation")
    plot_cols = st.columns([1, 1, 1])
    with plot_cols[0]:
        as_bars = st.toggle("Bars (counts par bin)", value=True, help="Affiche chaque ligne comme un bar chart (bins).")
    with plot_cols[1]:
        show_n = st.number_input("Nb d'histogrammes à afficher", min_value=1, max_value=max(1, len(edited_df)), value=min(3, len(edited_df)))
    with plot_cols[2]:
        start_idx = st.number_input("À partir de l'index (1-based)", min_value=1, max_value=len(edited_df), value=1)

    # Plot selected histograms
    subset = edited_df.iloc[start_idx - 1: start_idx - 1 + show_n]
    for idx, (row_label, row) in enumerate(subset.iterrows(), start=1):
        values = row.values.astype(float)
        finite_mask = ~np.isnan(values)
        values = values[finite_mask]
        fig = plt.figure()
        if as_bars:
            plt.bar(np.arange(1, len(values) + 1), values)
            plt.title(f"{row_label} — bars")
            plt.xlabel("Bin")
            plt.ylabel("Count")
        else:
            # If you wanted a true histogram from samples, you'd call plt.hist on samples.
            # Here we treat values as bin counts, so we still bar-plot for clarity.
            plt.bar(np.arange(1, len(values) + 1), values)
            plt.title(f"{row_label} — (counts)")
        st.pyplot(fig, use_container_width=True)

    st.divider()

    # =========================
    #       DIFF / COMPARE
    # =========================
    st.subheader("🧭 Comparator")
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    with col_comp1:
        if st.button("Diff vs Original (session)", use_container_width=True):
            dmask = diff_cells(st.session_state.df_original, st.session_state.df_current)
            if dmask.any().any():
                st.info("Cellules modifiées (True) :")
                st.dataframe(dmask, use_container_width=True)
                # Show a compact list of changes
                changes = []
                a, b = st.session_state.df_original.align(st.session_state.df_current, join="outer")
                for i in a.index.union(b.index):
                    for j in a.columns.union(b.columns):
                        va = a.at[i, j] if i in a.index and j in a.columns else np.nan
                        vb = b.at[i, j] if i in b.index and j in b.columns else np.nan
                        if not (pd.isna(va) and pd.isna(vb)) and not (va == vb):
                            changes.append((i, j, va, vb))
                if changes:
                    chdf = pd.DataFrame(changes, columns=["row", "col", "original", "current"])
                    st.dataframe(chdf, use_container_width=True, hide_index=True)
            else:
                st.success("Aucune différence avec l’original en session.")

    with col_comp2:
        if st.session_state.source_kind == "path" and st.session_state.source_path:
            if st.button("Diff vs Disque (actuel)", use_container_width=True):
                try:
                    disk_df, _, _ = load_from_path(st.session_state.source_path)
                    dmask2 = diff_cells(disk_df, st.session_state.df_current)
                    if dmask2.any().any():
                        st.info("Différences vs disque (True) :")
                        st.dataframe(dmask2, use_container_width=True)
                    else:
                        st.success("Pas de différence avec le fichier sur disque.")
                except Exception as e:
                    st.error(f"Erreur lecture disque: {e}")
        else:
            st.caption("Diff vs disque indisponible pour un fichier uploadé.")

    with col_comp3:
        if st.session_state.source_kind == "path" and st.session_state.source_path:
            if st.button("Conflits (3-way: original / current / disk)", use_container_width=True):
                try:
                    disk_df, _, _ = load_from_path(st.session_state.source_path)
                    conflicts = compute_conflicts(
                        st.session_state.df_original,
                        st.session_state.df_current,
                        disk_df,
                    )
                    if conflicts.any().any():
                        st.warning("Conflits détectés (True) :")
                        st.dataframe(conflicts, use_container_width=True)
                    else:
                        st.success("Aucun conflit 3-way.")
                except Exception as e:
                    st.error(f"Erreur lecture disque: {e}")
        else:
            st.caption("Conflits 3-way indisponibles pour un upload.")

    # =========================
    #          APPLY
    # =========================
    st.subheader("✅ Apply / Save")

    if st.session_state.source_kind == "path" and st.session_state.source_path:
        # Show external change detection
        with st.container(border=True):
            try:
                current_h, current_m = file_hash_and_mtime(st.session_state.source_path)
                changed_externally = (current_h != st.session_state.file_hash) or (current_m != st.session_state.file_mtime)
                if changed_externally:
                    st.warning("Le fichier sur disque a changé depuis le Load.")
                else:
                    st.info("Le fichier sur disque n'a pas changé depuis le Load.")
            except Exception:
                st.caption("Impossible de vérifier l’état du fichier sur disque.")

        strategy = st.selectbox(
            "Stratégie si le fichier a changé sur disque (3-way merge)",
            options=["ours", "theirs", "overwrite"],
            help=(
                "'ours' = garde tes modifs en cas de conflit; "
                "'theirs' = privilégie le disque en cas de conflit; "
                "'overwrite' = écrase entièrement avec tes données."
            ),
        )
        if st.button("Apply to disk", type="primary", use_container_width=True):
            try:
                if os.path.exists(st.session_state.source_path):
                    if strategy == "overwrite":
                        final_df = st.session_state.df_current
                    else:
                        disk_df, _, _ = load_from_path(st.session_state.source_path)
                        final_df = merge_3way(
                            st.session_state.df_original,
                            st.session_state.df_current,
                            disk_df,
                            strategy=strategy,
                        )
                    safe_write_to_path(st.session_state.source_path, final_df)
                    # refresh original baseline & hashes
                    st.session_state.df_original = final_df.copy()
                    h, m = file_hash_and_mtime(st.session_state.source_path)
                    st.session_state.file_hash = h
                    st.session_state.file_mtime = m
                    st.success(f"Écrit sur disque: {st.session_state.source_path}")
                else:
                    st.error("Le chemin n'existe plus.")
            except Exception as e:
                st.error(f"Erreur écriture: {e}")
    else:
        # Upload case: provide download
        out_text = df_to_lines(st.session_state.df_current)
        st.download_button(
            "💾 Download modified file",
            data=out_text.encode("utf-8"),
            file_name=(st.session_state.source_path or "histograms.txt").replace(".txt", "_edited.txt"),
            mime="text/plain",
            use_container_width=True,
        )

st.caption(
    f"Sessions actives (approx.): {registry.prune_and_count(SHUTDOWN_TIMEOUT)} • "
    f"Heartbeat: {HEARTBEAT_INTERVAL}s • Timeout: {SHUTDOWN_TIMEOUT}s"
)
