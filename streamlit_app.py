# streamlit_app.py
# Pick-4 (3389 / 3889 / 3899 families) prediction helper:
# - Master ranking: all streams (State/Game) scored for likelihood a family hits next
# - Straight ordering learner: per State/Game, rank the 12 unique straights for each family
#
# Notes:
# - Accepts BOTH .txt and .csv inputs.
# - TXT parsing is resilient to: tabs, multiple spaces, commas, Fireball/Wild Ball trailing text.
# - Playable list upload is OPTIONAL (marks PlayableByUser=Yes/No; never filters rows).

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------

st.set_page_config(page_title="Pick 4 3389/3889/3899 — Master Ranking + Straights", layout="wide")

FAMILY_KEYS = {
    "3389": "3389",
    "3889": "3889",
    "3899": "3899",
}
FAMILY_SET = set(FAMILY_KEYS.keys())

DIGIT_RE = re.compile(r"\d")


# -----------------------------
# Utilities
# -----------------------------


def _safe_decode(uploaded) -> str:
    """Read a Streamlit UploadedFile as text."""
    raw = uploaded.getvalue()
    # Try UTF-8 first; fall back to latin-1 to avoid hard failures.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def normalize_state(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_game(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_4digit_result(s: str) -> Optional[str]:
    """Extract a 4-digit result from a messy field.

    Handles patterns like:
      - 3-9-3-8
      - 3938
      - 3-9-3-8, Fireball: 9
      - ... Wild Ball: 2

    Returns a 4-char string or None.
    """
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None

    # Common: digits separated by dashes/spaces: "3-9-3-8"
    digits = DIGIT_RE.findall(txt)
    if len(digits) < 4:
        return None

    # IMPORTANT: We want the FIRST 4 digits describing the result, not trailing fireball.
    # Many lines append Fireball/Wild Ball digits after the result, so we take the first 4.
    return "".join(digits[:4])


def box_key(num4: str) -> str:
    return "".join(sorted(num4))


def unique_perms(num4: str) -> List[str]:
    return sorted({"".join(p) for p in permutations(list(num4), 4)})


def parse_date_any(s: str) -> Optional[pd.Timestamp]:
    """Parse date from common formats in your TXT/CSV."""
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None

    # Typical lines: "Sat, Dec 27, 2025" or "2025/12/26"
    # Let pandas try first.
    try:
        dt = pd.to_datetime(txt, errors="coerce", infer_datetime_format=True)
        if pd.isna(dt):
            return None
        return dt
    except Exception:
        return None


# -----------------------------
# Parsing (HITS + STREAM files)
# -----------------------------

REQUIRED_COLS = ["date", "state", "game", "result"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        mapping[c] = lc
    df = df.rename(columns=mapping)

    # Common aliases
    if "drawdate" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"drawdate": "date"})
    if "draw" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"draw": "date"})
    if "province" in df.columns and "state" not in df.columns:
        df = df.rename(columns={"province": "state"})

    return df


def _parse_csv_flexible(text: str) -> Optional[pd.DataFrame]:
    """Attempt CSV/TSV parsing using pandas.

    NOTE: many files that look like "text" are actually TSV where the
    date contains commas (e.g., "Fri, Dec 26, 2025"). If we let pandas
    sniff delimiters, it can wrongly choose comma and shred the date.
    So we try tab-first when tabs are present.
    """

    # Grab the first non-empty line for delimiter heuristics
    first_line = ""
    for ln in text.splitlines():
        if ln.strip():
            first_line = ln
            break

    # Build a small list of parsing attempts, ordered by likelihood.
    attempts: List[Dict[str, object]] = []
    if "\t" in first_line and first_line.count("\t") >= 2:
        attempts.append(dict(sep="\t", engine="python"))
    # Common cases
    attempts.extend([
        dict(sep=None, engine="python"),
        dict(sep=",", engine="python"),
        dict(sep=";", engine="python"),
    ])

    df = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(io.StringIO(text), **kwargs)
            break
        except Exception:
            df = None
            continue

    if df is None:
        return None
    df = _standardize_columns(df)

    # If it already contains required columns, great.
    if all(c in df.columns for c in REQUIRED_COLS):
        return df

    # Sometimes the file is delimiter-separated but without header.
    # Try heuristic: 4 columns in order date, state, game, result.
    if df.shape[1] >= 4 and "date" not in df.columns:
        cols = list(df.columns)
        df2 = df.rename(columns={cols[0]: "date", cols[1]: "state", cols[2]: "game", cols[3]: "result"})
        if all(c in df2.columns for c in REQUIRED_COLS):
            # Validate that at least one date is parseable; otherwise this
            # is likely a bad delimiter (e.g., comma split inside the date).
            sample = df2["date"].head(8).astype(str).tolist()
            if any(parse_date_any(x) is not None for x in sample):
                return df2

    return None


def _parse_txt_lines(text: str) -> pd.DataFrame:
    """Parse TXT where each line is like:

    Sat, Dec 27, 2025\tTexas\tDaily 4 Night\t3-9-3-8, Fireball: 9

    or with multiple spaces.
    """
    rows: List[Dict[str, object]] = []

    # Heuristic keywords that typically mark where the **Game** field begins
    # when a line is separated by single spaces (no tabs / no multi-spaces).
    # This helps parse lines like:
    # "Fri, Dec 26, 2025 Texas Daily 4 Night 3-9-3-8, Fireball: 9"
    GAME_KEYWORDS = {
        "pick", "daily", "cash", "numbers", "dc-4", "pega", "midday", "evening", "night", "day",
        "am", "pm", "a.m.", "p.m.", "10pm", "11pm", "7:50pm", "1:50pm",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Skip obvious headers
        low = line.lower()
        if low.startswith("date") and ("state" in low and "game" in low):
            continue

        # Prefer tab split, else 2+ spaces, else a keyword-based single-space fallback
        parts: List[str]
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) < 4:
                # Some lines are comma-separated in the date segment.
                # Try to capture date up to year, then remaining fields.
                m = re.match(r"^(.*?\b\d{4}\b)\s+(.*)$", line)
                if m:
                    date_part = m.group(1).strip()
                    rest = m.group(2).strip()
                    rest_parts = [p.strip() for p in re.split(r"\s{2,}", rest) if p.strip()]
                    parts = [date_part] + rest_parts

        if len(parts) < 4:
            # Keyword-based parse for single-space-separated lines.
            # Strategy:
            # 1) Grab date (up to year)
            # 2) Find the 4-digit result (with or without hyphens)
            # 3) Remaining middle => "State Game"; split into state vs game by first game keyword
            m = re.match(r"^(.*?\b\d{4}\b)\s+(.*)$", line)
            if m:
                date_part = m.group(1).strip()
                rest = m.group(2).strip()

                # Find first 4-digit pick4-like token in the rest (hyphenated or compact)
                rm = re.search(r"(\d\s*[- ]\s*\d\s*[- ]\s*\d\s*[- ]\s*\d|\b\d{4}\b)", rest)
                if rm:
                    mid = rest[: rm.start()].strip()
                    result_part = rest[rm.start():].strip()

                    # Split mid into tokens
                    toks = mid.split()
                    if len(toks) >= 2:
                        # Find first token that looks like a game keyword
                        gi = None
                        for i, t in enumerate(toks):
                            if t.lower() in GAME_KEYWORDS or t.lower().startswith("dc-"):
                                gi = i
                                break
                        if gi is not None and gi > 0:
                            state = " ".join(toks[:gi]).strip()
                            game = " ".join(toks[gi:]).strip()
                            parts = [date_part, state, game, result_part]

        if len(parts) < 4:
            # As an absolute fallback, try comma split (rare)
            parts = [p.strip() for p in line.split(",") if p.strip()]

        if len(parts) < 4:
            continue

        dt = parse_date_any(parts[0])
        if dt is None:
            continue

        state = parts[1]
        game = parts[2]
        result = parts[3]

        rows.append({"date": dt, "state": state, "game": game, "result": result})

    df = pd.DataFrame(rows)
    return df


def load_history(uploaded, filter_families: bool = True) -> pd.DataFrame:
    """Load either CSV or TXT into a canonical df with required columns."""
    if uploaded is None:
        return pd.DataFrame(columns=REQUIRED_COLS)

    name = (uploaded.name or "").lower()
    text = _safe_decode(uploaded)

    df: Optional[pd.DataFrame] = None
    if name.endswith(".csv"):
        df = _parse_csv_flexible(text)
        if df is None:
            # Sometimes CSVs are weird; try txt parser
            df = _parse_txt_lines(text)
    elif name.endswith(".txt"):
        # Many of your "txt" files are actually tab-separated tables.
        # Also: dates like "Fri, Dec 26, 2025" contain commas, so pandas'
        # delimiter sniffing can mistakenly pick comma and shred the date.
        # Prefer the robust line parser first.
        df = _parse_txt_lines(text)
        if df is None:
            df = _parse_csv_flexible(text)
    else:
        # Unknown extension: try delimiter sniff then txt parser
        df = _parse_csv_flexible(text)
        if df is None:
            df = _parse_txt_lines(text)

    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    df = _standardize_columns(df)

    # Keep only required columns (but tolerate extra)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = None

    # Normalize
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["state"] = df["state"].astype(str).map(normalize_state)
    df["game"] = df["game"].astype(str).map(normalize_game)
    df["result"] = df["result"].map(extract_4digit_result)

    df = df.dropna(subset=["date", "state", "game", "result"]).copy()
    df["result"] = df["result"].astype(str)
    # Attach family key
    df["box"] = df["result"].map(box_key)
    df["family"] = df["box"].where(df["box"].isin(FAMILY_SET), other="")
    if filter_families:
        df = df[df["family"] != ""].copy()
    df["family"] = df["family"].astype(str)

    # Sort
    df = df.sort_values(["state", "game", "date"]).reset_index(drop=True)
    return df


def load_playable_list(uploaded) -> pd.DataFrame:
    """Load playable streams list. Accepts TXT or CSV.

    Must include State and Game either as header or as 2 columns.
    """
    if uploaded is None:
        return pd.DataFrame(columns=["state", "game"])

    text = _safe_decode(uploaded)
    name = (uploaded.name or "").lower()

    df: Optional[pd.DataFrame] = None
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            df = None
    if df is None:
        # TXT or fallback: parse as simple lines
        rows = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("state") and "game" in low:
                continue
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t") if p.strip()]
            else:
                # try comma, else 2+ spaces
                if "," in line:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                else:
                    parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) < 2:
                continue
            rows.append({"state": normalize_state(parts[0]), "game": normalize_game(parts[1])})
        df = pd.DataFrame(rows)

    df = _standardize_columns(df)
    # Accept either state/game or State/Game
    if "state" not in df.columns or "game" not in df.columns:
        # If exactly 2 cols, map them
        if df.shape[1] >= 2:
            cols = list(df.columns)
            df = df.rename(columns={cols[0]: "state", cols[1]: "game"})

    if "state" not in df.columns or "game" not in df.columns:
        return pd.DataFrame(columns=["state", "game"])

    df = df[["state", "game"]].copy()
    df["state"] = df["state"].astype(str).map(normalize_state)
    df["game"] = df["game"].astype(str).map(normalize_game)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df


# -----------------------------
# Scoring (Master Ranking)
# -----------------------------


def exp_decay(days: float, half_life: float) -> float:
    if half_life <= 0:
        return 0.0
    return float(math.exp(-math.log(2) * (days / half_life)))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


@dataclass
class StreamStats:
    state: str
    game: str
    hits: int
    last_hit_date: Optional[pd.Timestamp]
    days_since_last_hit: Optional[int]
    hit_rate: float
    consistency: float
    reliability: float
    share_3389: float
    share_3889: float
    share_3899: float
    overdue_percentile: float
    due_tempered: float
    schedule_boost: float
    expected_gap_days: Optional[float]
    predicted_next_hit: Optional[pd.Timestamp]


def compute_schedule_boost(dates: Sequence[pd.Timestamp],
                           target: pd.Timestamp,
                           alpha: float,
                           combine_mode: str) -> float:
    if len(dates) == 0:
        return 0.0

    # weekday: 0=Mon..6=Sun
    wds = [int(d.weekday()) for d in dates]
    months = [int(d.month) for d in dates]

    total = len(dates)
    wd_counts = np.bincount(wds, minlength=7)
    mo_counts = np.bincount(months, minlength=13)  # 1..12 used

    wd_prob = safe_div(wd_counts[target.weekday()] + alpha, total + alpha * 7)
    mo_prob = safe_div(mo_counts[target.month] + alpha, total + alpha * 12)

    if combine_mode.lower().startswith("mult"):
        return float(wd_prob * mo_prob)
    # default: average
    return float((wd_prob + mo_prob) / 2.0)


def compute_stream_stats(df: pd.DataFrame,
                         analysis_start: pd.Timestamp,
                         analysis_end: pd.Timestamp,
                         schedule_alpha: float,
                         schedule_combine_mode: str) -> pd.DataFrame:
    """Compute stats per (state, game) using ONLY the hits history."""

    groups = df.groupby(["state", "game"], sort=False)
    rows = []

    total_days = max(1, int((analysis_end.date() - analysis_start.date()).days) + 1)

    for (state, game), g in groups:
        g = g.sort_values("date")
        hits = int(len(g))

        last_hit_date = g["date"].max() if hits else None
        days_since = None
        if last_hit_date is not None:
            days_since = int((analysis_end.normalize() - last_hit_date.normalize()).days)

        # HitRate: hits per day in window
        hit_rate = safe_div(hits, total_days)

        # Consistency: fraction of months in window with >=1 hit
        if hits:
            months_with = g["date"].dt.to_period("M").nunique()
        else:
            months_with = 0
        months_total = max(1, (analysis_end.to_period("M") - analysis_start.to_period("M")).n + 1)
        consistency = safe_div(months_with, months_total)

        # Reliability: soft boost for sample size
        reliability = math.log1p(hits)

        # Family shares
        counts_by_family = g["family"].value_counts().to_dict()
        share_3389 = safe_div(counts_by_family.get("3389", 0), hits)
        share_3889 = safe_div(counts_by_family.get("3889", 0), hits)
        share_3899 = safe_div(counts_by_family.get("3899", 0), hits)

        # Gap-based overdue and expected next hit
        dates = g["date"].sort_values().dt.normalize().tolist()
        gaps = []
        for i in range(1, len(dates)):
            gap = int((dates[i] - dates[i - 1]).days)
            if gap > 0:
                gaps.append(gap)

        overdue_pct = 0.0
        expected_gap = None
        predicted_next = None
        due_tempered = 0.0

        if gaps and days_since is not None:
            gaps_arr = np.array(gaps, dtype=float)
            expected_gap = float(np.mean(gaps_arr))

            # OverduePercentile: how deep are we relative to historical gaps
            overdue_pct = float(np.mean(gaps_arr <= float(days_since)))

            # Temper by proximity to median-ish gaps (avoid over-weighting extreme droughts)
            med = float(np.median(gaps_arr))
            prox = math.exp(-abs(float(days_since) - med) / (med + 1.0))
            due_tempered = float(overdue_pct * (0.25 + 0.75 * prox))

            # Predicted next hit date
            if last_hit_date is not None:
                predicted_next = last_hit_date.normalize() + pd.Timedelta(days=int(round(expected_gap)))

        schedule_boost = compute_schedule_boost(dates, analysis_end.normalize(), alpha=schedule_alpha,
                                                combine_mode=schedule_combine_mode)

        rows.append({
            "State": state,
            "Game": game,
            "Hits": hits,
            "LastHitDate": last_hit_date.date().isoformat() if last_hit_date is not None else "",
            "DaysSinceLastHit": days_since if days_since is not None else "",
            "HitRate": hit_rate,
            "Consistency": consistency,
            "Reliability": reliability,
            "Share_3389": share_3389,
            "Share_3889": share_3889,
            "Share_3899": share_3899,
            "OverduePercentile": overdue_pct,
            "DueTempered": due_tempered,
            "ScheduleBoost": schedule_boost,
            "ExpectedGapDays": expected_gap if expected_gap is not None else "",
            "PredictedNextHitDate": predicted_next.date().isoformat() if predicted_next is not None else "",
        })

    return pd.DataFrame(rows)


def score_master_table(stats: pd.DataFrame,
                       w_hit: float,
                       w_due: float,
                       w_sched: float,
                       w_cons: float,
                       w_rel: float) -> pd.DataFrame:
    if stats.empty:
        return stats

    df = stats.copy()

    # Normalize components to comparable 0..1 range
    def minmax(col: str) -> pd.Series:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        mn, mx = float(x.min()), float(x.max())
        if mx <= mn:
            return pd.Series([0.0] * len(x), index=x.index)
        return (x - mn) / (mx - mn)

    hit_n = minmax("HitRate")
    due_n = minmax("DueTempered")
    sched_n = minmax("ScheduleBoost")
    cons_n = minmax("Consistency")
    rel_n = minmax("Reliability")

    score = (
        w_hit * hit_n
        + w_due * due_n
        + w_sched * sched_n
        + w_cons * cons_n
        + w_rel * rel_n
    )

    df["Score"] = score
    df = df.sort_values(["Score", "Hits"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


# -----------------------------
# Straights learner (per stream)
# -----------------------------


def straight_ranking_for_stream(df_stream: pd.DataFrame,
                               state: str,
                               game: str,
                               family: str,
                               alpha: float = 1.0,
                               half_life: int = 120,
                               recency_mix: float = 0.30,
                               asof: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, Dict]:
    """Return a 12-straight ranking table for one (state, game, family).

    If the selected stream has 0 hits for this family, we fall back to a global
    distribution (all uploaded stream rows) for that family. If that is also 0,
    we use a uniform distribution.

    Returns: (table_df, info_dict)
    """
    perms = generate_unique_perms(family)

    if asof is None:
        asof = pd.Timestamp(df_stream["date"].max()).normalize() if not df_stream.empty else pd.Timestamp.today().normalize()
    else:
        asof = pd.Timestamp(asof).normalize()

    # Evidence: state/game specific hits for this family
    g = df_stream[(df_stream["state"] == state) & (df_stream["game"] == game) & (df_stream["family"] == family)].copy()
    used_mode = "state_game"

    # Global fallback evidence: any stream hits for this family
    g_global = df_stream[df_stream["family"] == family].copy() if not df_stream.empty else pd.DataFrame(columns=df_stream.columns if not df_stream.empty else [])

    if g.empty:
        if not g_global.empty:
            g = g_global
            used_mode = "global_fallback"
        else:
            used_mode = "uniform"

    # Counts
    counts = {p: 0 for p in perms}
    last_seen_map: dict[str, pd.Timestamp] = {}

    if used_mode != "uniform":
        # result is already canonical as 4 digits (e.g., '9383')
        for p, sub in g.groupby("result"):
            if p in counts:
                counts[p] = int(len(sub))
                last_seen_map[p] = pd.Timestamp(sub["date"].max()).normalize()

    total = sum(counts.values())

    # Freq prob (Laplace)
    denom = (total + alpha * len(perms))
    freq_prob = {p: (counts[p] + alpha) / denom for p in perms}

    # Recency weight per permutation
    rec_w = {}
    for p in perms:
        if p in last_seen_map:
            ds = (asof - last_seen_map[p]).days
            rec_w[p] = recency_weight(ds, half_life)
        else:
            rec_w[p] = 0.0

    # Normalize recency weights to look like a probability distribution
    rec_sum = sum(rec_w.values())
    if rec_sum > 0:
        rec_prob = {p: rec_w[p] / rec_sum for p in perms}
    else:
        rec_prob = {p: 1.0 / len(perms) for p in perms}

    # Final blend
    out_rows = []
    for p in perms:
        score = (1.0 - recency_mix) * freq_prob[p] + recency_mix * rec_prob[p]
        out_rows.append({
            "Straight": p,
            "Count": counts[p],
            "Prob": float(score),
        })

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["Prob", "Count", "Straight"], ascending=[False, False, True]).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))

    info = {
        "mode": used_mode,
        "state_game_hits": int(df_stream[(df_stream["state"] == state) & (df_stream["game"] == game) & (df_stream["family"] == family)].shape[0]) if not df_stream.empty else 0,
        "global_hits": int(g_global.shape[0]) if not df_stream.empty else 0,
        "asof": asof,
        "family": family,
        "state": state,
        "game": game,
    }

    return out, info



# -----------------------------
# UI
# -----------------------------

st.title("Pick 4 3389 / 3889 / 3899 — Master Ranking + 12-Straights Learner")

with st.sidebar:
    st.header("Inputs")
    st.caption("HITS = 5-year hits list across all states/games. STREAM = per-state/game 24-month history for straight ordering.")

    hits_file = st.file_uploader(
        "Upload 5-year HIT history (TXT or CSV)",
        type=["txt", "csv"],
        help="TXT can be tab-separated or space-separated. Must contain Date, State, Game, Result. Fireball/Wild Ball is okay.",
    )

    st.markdown("---")
    st.subheader("As-Of scoring date")
    asof_date = st.date_input("As-Of Date", value=None)
    assume_no_hits_after = st.checkbox(
        "Assume there were NO hits after the last file date through As-Of Date",
        value=True,
        help="If checked, days-since-last-hit is computed up to As-Of Date even when your file stops earlier.",
    )

    st.markdown("---")
    st.subheader("Optional: Playable list")
    playable_file = st.file_uploader(
        "Upload a Playable list (TXT or CSV with columns State,Game) to MARK playable streams (no filtering)",
        type=["txt", "csv"],
        help="If TXT: each line should be State<TAB>Game or State,Game.",
    )

    st.markdown("---")
    st.subheader("Model controls")
    # Fixed weights (but we still expose them as read-only text + hidden slider option)
    st.caption("Weights are fixed to your approved mix (you can change later if you want).")

    schedule_alpha = st.slider(
        "Schedule smoothing α (higher = weaker schedule boost / less overfit)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
    )
    schedule_mode = st.selectbox(
        "ScheduleBoost combine mode",
        options=["Multiply (weekday*month)", "Average (weekday+month)/2"],
        index=0,
    )

    # Default fixed weights (match your on-screen mix style)
    w_hit = 0.50
    w_due = 0.30
    w_sched = 0.10
    w_cons = 0.08
    w_rel = 0.02

    st.markdown("**Fixed scoring weights**")
    st.write(f"- {w_hit:.2f} HitRate")
    st.write(f"- {w_due:.2f} OverduePercentile (tempered by GapProximity)")
    st.write(f"- {w_sched:.2f} ScheduleBoost")
    st.write(f"- {w_cons:.2f} Consistency")
    st.write(f"- {w_rel:.2f} Reliability")

    st.markdown("---")
    st.subheader("Straights learning")
    stream_files = st.file_uploader(
        "Upload 24-month STREAM history file(s) (TXT or CSV)",
        type=["txt", "csv"],
        accept_multiple_files=True,
        help="You can upload one per state/game, or many at once. Must contain Date, State, Game, Result.",
    )

    straight_half_life = st.slider("Recency half-life (days)", 1, 365, 120)
    straight_alpha = st.slider("Smoothing alpha (Laplace)", 0.0, 5.0, 1.0, 0.05)
    straight_mix = st.slider("Blend recency vs frequency (0=freq only, 1=recency only)", 0.0, 1.0, 0.30, 0.01)


# ---------------
# Load data
# ---------------

if hits_file is None:
    st.info("Upload your 5-year HIT history to see the master ranking.")
    st.stop()

hits_df = load_history(hits_file, filter_families=True)

if hits_df.empty:
    st.error("After parsing, the HITS file contains 0 rows matching families 3389/3889/3899.")
    st.stop()

file_start = hits_df["date"].min().normalize()
file_end = hits_df["date"].max().normalize()

# Default As-Of = file end (or user override)
if asof_date is None:
    asof_ts = file_end
else:
    asof_ts = pd.Timestamp(asof_date)

analysis_end = asof_ts.normalize() if assume_no_hits_after else file_end
analysis_start = file_start

combine_mode = "multiply" if schedule_mode.lower().startswith("multiply") else "average"

st.caption(
    f"History USED: {analysis_start.date()} → {file_end.date()} (file end) | "
    f"As-Of scoring date: {asof_ts.date()} | Analysis window end: {analysis_end.date()} | "
    f"days_window={(analysis_end.date() - analysis_start.date()).days + 1} | "
    f"streams found: {hits_df.groupby(['state','game']).ngroups}"
)

# Playable list
playable_df = load_playable_list(playable_file) if playable_file is not None else pd.DataFrame(columns=["state", "game"])
playable_set = set()
if not playable_df.empty:
    playable_set = set(zip(playable_df["state"], playable_df["game"]))

# Compute per-stream stats + score
stats_df = compute_stream_stats(
    hits_df,
    analysis_start=analysis_start,
    analysis_end=analysis_end,
    schedule_alpha=float(schedule_alpha),
    schedule_combine_mode=combine_mode,
)

scored = score_master_table(
    stats_df,
    w_hit=w_hit,
    w_due=w_due,
    w_sched=w_sched,
    w_cons=w_cons,
    w_rel=w_rel,
)

if not scored.empty:
    scored["PlayableByUser"] = np.where(
        scored.apply(lambda r: (r["State"], r["Game"]) in playable_set, axis=1),
        "Yes",
        "No",
    )

# -----------------------------
# Master ranking display
# -----------------------------

st.header("A) Master Ranking — All States / All Games (Most → Least Likely)")

# Show a compact explanation
with st.expander("What this score is doing (quick)", expanded=False):
    st.markdown(
        """
- **HitRate**: streams that hit these families more often (in your file window) score higher.
- **OverduePercentile (tempered)**: streams that are **"due"** based on their own historical *hit gaps* score higher.
- **ScheduleBoost**: boosts streams that historically hit more often on the **same weekday/month** as the As-Of date.
- **Consistency / Reliability**: small stabilizers so tiny samples don’t dominate.

This is *not* using the winning 12/27 results directly — it is using hit-gap behavior learned from the history you uploaded.
        """
    )

st.dataframe(scored, use_container_width=True, height=520)

csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download master ranking CSV",
    data=csv_bytes,
    file_name="pk4_master_ranking_3389_3889_3899.csv",
    mime="text/csv",
)

# -----------------------------
# Straights learner display
# -----------------------------

st.header("B) Straight ordering learning (state-specific 12 straights per family)")

if not stream_files:
    st.info("Upload one or more 24-month STREAM history files (TXT or CSV) to rank straights per State/Game.")
    st.stop()

# Load and combine stream files
stream_dfs = []
for f in stream_files:
    d = load_history(f, filter_families=False)
    if not d.empty:
        stream_dfs.append(d)

if not stream_dfs:
    st.error("After parsing, the STREAM file(s) contain 0 rows. Please check formatting.")
    st.stop()

stream_df = pd.concat(stream_dfs, ignore_index=True)
stream_df = stream_df.sort_values(["state", "game", "date"]).reset_index(drop=True)

streams = sorted(set(zip(stream_df["state"], stream_df["game"])))

# Provide dropdown with State — Game
options = [f"{s} — {g}" for s, g in streams]
sel = st.selectbox("Pick a State/Game to rank straights:", options=options, index=0)
sel_state, sel_game = streams[options.index(sel)]

cols = st.columns(3)

for i, fam in enumerate(["3389", "3889", "3899"]):
    with cols[i]:
        st.subheader(f"{fam} — 12 straights")
        tbl = straight_ranking_for_stream(
            stream_df,
            state=sel_state,
            game=sel_game,
            family=fam,
            half_life_days=float(straight_half_life),
            alpha=float(straight_alpha),
            recency_mix=float(straight_mix),
            asof=analysis_end,
        )
        # If no local evidence, warn and show what fallback was used
        evid = str(tbl["Evidence"].iloc[0]) if len(tbl) else ""
        if evid != "state_game":
            if evid == "global_fallback":
                st.warning("No hits for this family in this State/Game within the uploaded window. Using GLOBAL (all uploaded streams) ordering as a fallback.")
            else:
                st.warning("No hits for this family found anywhere in the uploaded stream data. Showing an uninformed (uniform) ranking.")
        st.dataframe(tbl, use_container_width=True, height=480)

# Also allow export of current stream + tables
st.markdown("---")

# Bundle export: one CSV with all 36 rows for selected stream
all_rows = []
for fam in ["3389", "3889", "3899"]:
    tbl = straight_ranking_for_stream(
        stream_df,
        state=sel_state,
        game=sel_game,
        family=fam,
        half_life_days=float(straight_half_life),
        alpha=float(straight_alpha),
        recency_mix=float(straight_mix),
        asof=analysis_end,
    )
    tbl.insert(0, "Family", fam)
    tbl.insert(1, "State", sel_state)
    tbl.insert(2, "Game", sel_game)
    all_rows.append(tbl)

bundle = pd.concat(all_rows, ignore_index=True)

st.download_button(
    "Download straight ranking CSV for selected State/Game",
    data=bundle.to_csv(index=False).encode("utf-8"),
    file_name=f"pk4_straight_ranking_{sel_state}_{sel_game}.csv".replace(" ", "_").replace("/", "-"),
    mime="text/csv",
)
