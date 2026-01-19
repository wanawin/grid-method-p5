from __future__ import annotations

import io
import pathlib
import re
import math
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations_with_replacement, product
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set

import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning, message='invalid decimal literal')



# =========================================================
# Pick-5 SeedGrid + Dynamic Walk-Forward + Dynamic Filters
#
# Goals (as requested):
# 1) Reduce to ~450-500 candidates using grid+percentile-style ranking
# 2) Auto walk-forward choose best windows (120/180/240 learn; 60/90 validate)
# 3) Incorporate Loser List filters + Batch10 filters (dedupe redundant)
# 4) Dynamically score & rank filters per history file (per stream)
# 5) Apply filters automatically in safe->aggressive order until target
# 6) Final list ordered most-likely->least-likely AND emit straight-order list
#
# Notes:
# - This app is stream-agnostic. It expects a history file with one result per line.
# - It supports LotteryPost-like lines: "Sat, Dec 27, 2025    46894"
# - Winner checks are BOX-based for filters; straight list is derived at the end.
# =========================================================


st.set_page_config(page_title="Pick 5 SeedGrid + Walk-Forward (Dynamic)", layout="wide")


DIGITS = list("0123456789")
MIRROR = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}

# Keep DC-5-style mappings for Batch10 compatibility
V_TRAC_GROUPS = {0: 1, 5: 1, 1: 2, 6: 2, 2: 3, 7: 3, 3: 4, 8: 4, 4: 5, 9: 5}
V_TRAC = V_TRAC_GROUPS
VTRAC_GROUPS = V_TRAC_GROUPS
vtrac = V_TRAC_GROUPS
mirror = MIRROR


# -------------------------
# Parsing history
# -------------------------

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _normalize_quotes(text: str) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\r\n", "\n").replace("\r", "\n")
    )


def parse_date_any(line: str) -> Optional[datetime]:
    """Attempt to parse a date from many common history formats."""
    s = line.strip()

    # mm/dd/yyyy or m/d/yy
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", s)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yy < 100:
            yy += 2000
        try:
            return datetime(yy, mm, dd)
        except Exception:
            return None

    # LotteryPost style: "Sat, Dec 27, 2025"
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\b", s)
    if m:
        mon = MONTHS.get(m.group(1).lower())
        if mon:
            dd, yy = int(m.group(2)), int(m.group(3))
            try:
                return datetime(yy, mon, dd)
            except Exception:
                return None

    # yyyy-mm-dd
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)
    if m:
        yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(yy, mm, dd)
        except Exception:
            return None

    return None


def extract_last_5digit_result(line: str) -> Optional[str]:
    """Return the last Pick-5 style result on the line.

    History sources use multiple formats. We support:
      - contiguous: 46894
      - separated: 4-6-8-9-4, 4 6 8 9 4, 4,6,8,9,4
    We intentionally *avoid* manufacturing results from date fragments.
    """

    # 1) Common case: a standalone 5-digit token
    # Use digit-boundaries (not word-boundaries) to handle cases like "46894," or "46894)".
    nums = re.findall(r"(?<!\d)\d{5}(?!\d)", line)
    if nums:
        return nums[-1]

    # 2) Hyphen/space/comma separated digits: 4-6-8-9-4, 4 6 8 9 4, etc.
    # Require separators between digits to prevent accidental captures from years.
    parts = re.findall(r"(?<!\d)(\d)\D+(\d)\D+(\d)\D+(\d)\D+(\d)(?!\d)", line)
    if parts:
        d1, d2, d3, d4, d5 = parts[-1]
        return f"{d1}{d2}{d3}{d4}{d5}"

    return None


def load_history_from_text(text: str) -> pd.DataFrame:
    """Returns DataFrame with columns: date (datetime or NaT), result (str)."""
    text = _normalize_quotes(text or "")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    rows = []
    for ln in lines:
        res = extract_last_5digit_result(ln)
        if not res:
            continue
        dt = parse_date_any(ln)
        rows.append({"date": dt, "result": res, "raw": ln})
    if not rows:
        raise ValueError("No 5-digit results found in the uploaded history.")

    df = pd.DataFrame(rows)

    # If no dates at all, preserve order as-is (assume already MR→Oldest)
    if df["date"].notna().sum() == 0:
        df["idx"] = range(len(df))
        # assume top of file is most recent
        df = df.sort_values("idx", ascending=True).drop(columns=["idx"]).reset_index(drop=True)
        return df

    # Otherwise sort by date descending; stable tie-breaker by appearance order
    df["idx"] = range(len(df))
    df = df.sort_values(["date", "idx"], ascending=[False, True]).drop(columns=["idx"]).reset_index(drop=True)
    return df


# -------------------------
# Grid + candidate generation
# -------------------------


def digits_of(nstr: str) -> List[int]:
    return [int(c) for c in nstr]


def mirror_digits(ds: Iterable[int]) -> List[int]:
    return [MIRROR[int(d)] for d in ds]


def plus1_digits(ds: Iterable[int]) -> List[int]:
    return [(int(d) + 1) % 10 for d in ds]


def minus1_digits(ds: Iterable[int]) -> List[int]:
    return [(int(d) - 1) % 10 for d in ds]


def make_grid_sets(seed: str) -> Dict[str, List[int]]:
    s = digits_of(seed)
    return {
        "seed": s,
        "+1": plus1_digits(s),
        "-1": minus1_digits(s),
        "mirror": mirror_digits(s),
    }


def box_key_from_digits(ds: List[int]) -> str:
    return "".join(str(d) for d in sorted(ds))


def build_pair_pool(digits_set: Set[int]) -> List[Tuple[int, int]]:
    ds = sorted(digits_set)
    return list(combinations_with_replacement(ds, 2))


def generate_candidate_boxes(
    seed: str,
    mode: str = "two_pairs_plus_non_grid",
    cap: int = 20000,
    rng_seed: int = 0,
) -> List[str]:
    """Generate box-unique 5-digit combos (as sorted string) with repeats allowed."""
    grid = make_grid_sets(seed)
    grid_digits = set(grid["seed"]) | set(grid["+1"]) | set(grid["-1"]) | set(grid["mirror"])
    non_grid_digits = set(range(10)) - set(grid_digits)

    # pair pools
    seed_plus = set(grid["seed"]) | set(grid["+1"]) | set(grid["-1"])
    pair_seed = build_pair_pool(seed_plus)
    pair_grid = build_pair_pool(grid_digits)
    pair_mirror = build_pair_pool(set(grid["mirror"]))

    # Use deterministic ordering (no random) to keep runs stable.
    out: Set[str] = set()

    def add_box(digits5: List[int]):
        out.add(box_key_from_digits(digits5))

    # Mode choices:
    # - two_pairs_plus_non_grid: (seed-pair) + (grid-pair) + (one non-grid digit)
    # - two_pairs_plus_grid:     (seed-pair) + (grid-pair) + (one grid digit)
    # - mirror_pair_mix:         (seed-pair) + (mirror-pair) + (one non-grid digit)
    if mode == "two_pairs_plus_non_grid":
        fifth_pool = sorted(non_grid_digits) if non_grid_digits else sorted(grid_digits)
        for a in pair_seed:
            for b in pair_grid:
                for c in fifth_pool:
                    add_box([a[0], a[1], b[0], b[1], c])
                    if len(out) >= cap:
                        return sorted(out)

    elif mode == "two_pairs_plus_grid":
        fifth_pool = sorted(grid_digits)
        for a in pair_seed:
            for b in pair_grid:
                for c in fifth_pool:
                    add_box([a[0], a[1], b[0], b[1], c])
                    if len(out) >= cap:
                        return sorted(out)

    elif mode == "mirror_pair_mix":
        fifth_pool = sorted(non_grid_digits) if non_grid_digits else sorted(grid_digits)
        for a in pair_seed:
            for b in pair_mirror:
                for c in fifth_pool:
                    add_box([a[0], a[1], b[0], b[1], c])
                    if len(out) >= cap:
                        return sorted(out)
    else:
        raise ValueError(f"Unknown generation mode: {mode}")

    return sorted(out)


# -------------------------
# Dynamic stats (hot/cold/due etc)
# -------------------------


def last_n_draws(df: pd.DataFrame, start_idx: int, n: int) -> List[str]:
    """From df sorted MR->Oldest: start_idx is seed index, return df[start_idx : start_idx+n]."""
    return df.iloc[start_idx:start_idx + n]["result"].tolist()


def digit_freq(draws: List[str]) -> Counter:
    c = Counter()
    for r in draws:
        c.update(list(r))
    for d in DIGITS:
        c.setdefault(d, 0)
    return c


def hot_cold_sets(draws: List[str], hot_k: int = 3, cold_k: int = 3) -> Tuple[List[int], List[int]]:
    c = digit_freq(draws)
    # hot: highest count, then digit asc
    hot = sorted(DIGITS, key=lambda d: (-c[d], int(d)))[:hot_k]
    cold = sorted(DIGITS, key=lambda d: (c[d], int(d)))[:cold_k]
    return [int(x) for x in hot], [int(x) for x in cold]


def due_last2(prev: str, prev2: str) -> List[int]:
    miss = set(DIGITS) - set(prev) - set(prev2)
    return sorted([int(x) for x in miss])


def sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return 'Very Low'
    if 16 <= total <= 24:
        return 'Low'
    if 25 <= total <= 33:
        return 'Mid'
    return 'High'


def structure_of(digits: List[int]) -> str:
    counts = sorted(Counter(digits).values(), reverse=True)
    if counts == [1, 1, 1, 1, 1]:
        return 'SINGLE'
    if counts == [2, 1, 1, 1]:
        return 'DOUBLE'
    if counts == [2, 2, 1]:
        return 'DOUBLE-DOUBLE'
    if counts == [3, 1, 1]:
        return 'TRIPLE'
    if counts == [3, 2]:
        return 'TRIPLE-DOUBLE'
    if counts == [4, 1]:
        return 'QUAD'
    if counts == [5]:
        return 'QUINT'
    return f'OTHER-{counts}'


# -------------------------
# Scoring candidates (grid + learned tendencies)
# -------------------------


@dataclass
class LearnedTendencies:
    hot_digits: List[int]
    cold_digits: List[int]
    due_bias_digits: List[int]
    # seed-winner tendencies (learned)
    typical_seed_overlap: List[int]  # preferred overlaps (e.g., [1,2])
    structure_weights: Dict[str, float]
    sum_mean: float
    sum_sd: float


def learn_tendencies_from_transitions(
    transitions: List[Tuple[str, str]],
    hotcold_draws: List[str],
) -> LearnedTendencies:
    """Learn broad tendencies from (seed, next_winner) transitions."""
    # Hot/cold from recent context (last 20 by default outside)
    hot, cold = hot_cold_sets(hotcold_draws, hot_k=3, cold_k=3)

    overlap_counts = []
    struct_counts = Counter()
    sums = []
    due_hits = Counter()

    for seed, nxt in transitions:
        sset = set(seed)
        o = len(set(nxt) & sset)
        overlap_counts.append(o)
        digs = [int(x) for x in nxt]
        struct_counts[structure_of(digs)] += 1
        sums.append(sum(digs))

        # due digits relative to the seed and its prev2 are unknown here; we approximate
        # due tendency as "digits that appear less often overall" within this window
        for d in nxt:
            due_hits[int(d)] += 1

    # typical overlap: take the two most common overlap counts
    if overlap_counts:
        oc = Counter(overlap_counts)
        typical = [k for k, _ in oc.most_common(2)]
    else:
        typical = [1, 2]

    # structure weights: normalize counts into weights; favor common structures
    total = sum(struct_counts.values()) or 1
    weights = {k: v / total for k, v in struct_counts.items()}

    mu = float(sum(sums) / len(sums)) if sums else 22.5
    var = float(sum((x - mu) ** 2 for x in sums) / max(1, (len(sums) - 1))) if len(sums) > 1 else 30.0
    sd = math.sqrt(var) if var > 0 else 5.0

    # crude due-bias digits: digits with lower hit frequency become "due-biased"
    if due_hits:
        min_hits = min(due_hits.values())
        due_bias = sorted([d for d in range(10) if due_hits[d] <= min_hits + 1])
    else:
        due_bias = [0, 1, 2, 3]

    return LearnedTendencies(
        hot_digits=hot,
        cold_digits=cold,
        due_bias_digits=due_bias,
        typical_seed_overlap=typical,
        structure_weights=weights,
        sum_mean=mu,
        sum_sd=sd,
    )


def candidate_score(box: str, seed: str, grid_digits: Set[int], tend: LearnedTendencies) -> float:
    ds = [int(c) for c in box]
    counts = Counter(ds)
    unique = len(counts)
    s_overlap = len(set(seed) & set(box))
    ssum = sum(ds)

    score = 0.0

    # 1) Grid inclusion boosts
    grid_hits = sum(1 for d in ds if d in grid_digits)
    score += 0.6 * grid_hits

    # 2) Seed overlap preference
    score += 1.2 if s_overlap in set(tend.typical_seed_overlap) else 0.0
    score += 0.25 * s_overlap

    # 3) Hot/cold presence (soft)
    score += 0.35 * sum(1 for d in ds if d in tend.hot_digits)
    score += 0.20 * sum(1 for d in ds if d in tend.cold_digits)
    score += 0.10 * sum(1 for d in ds if d in tend.due_bias_digits)

    # 4) Structure likelihood
    struct = structure_of(ds)
    score += 2.0 * float(tend.structure_weights.get(struct, 0.0))

    # 5) Sum proximity (Gaussian-ish)
    z = abs(ssum - tend.sum_mean) / max(1e-9, tend.sum_sd)
    score += max(0.0, 1.0 - 0.25 * z)

    # 6) Mild penalty for extreme repetition
    if unique <= 2:
        score -= 0.8
    if unique == 1:
        score -= 1.5

    return float(score)


def rank_candidates(boxes: List[str], seed: str, tend: LearnedTendencies) -> List[Tuple[str, float]]:
    grid = make_grid_sets(seed)
    grid_digits = set(grid["seed"]) | set(grid["+1"]) | set(grid["-1"]) | set(grid["mirror"])
    scored = [(b, candidate_score(b, seed, grid_digits, tend)) for b in boxes]
    # stable tie-break by numeric value
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


# -------------------------
# Dynamic Percentile Zones (winner-heavy bins)
# -------------------------


def percentile_bin(idx: int, n: int, bin_size: int = 5) -> int:
    if n <= 0:
        return 0
    pct = (idx / max(1, (n - 1))) * 100.0
    return int(pct // bin_size) * bin_size  # 0,5,10,...


def compute_winner_heavy_bins(
    ranked_lists: List[List[str]],
    winners: List[str],
    bin_size: int = 5,
    keep_adjacent: bool = True,
) -> Set[int]:
    """Return set of percentile bins (0..95) that contain at least one winner."""
    heavy = set()
    for ranked, win in zip(ranked_lists, winners):
        if not ranked:
            continue
        win_box = "".join(sorted(win))
        try:
            idx = ranked.index(win_box)
        except ValueError:
            continue
        b = percentile_bin(idx, len(ranked), bin_size=bin_size)
        heavy.add(b)
        if keep_adjacent:
            heavy.add(max(0, b - bin_size))
            heavy.add(min(100 - bin_size, b + bin_size))
    return heavy


def apply_percentile_bins(ranked: List[Tuple[str, float]], bins: Set[int], bin_size: int = 5) -> List[Tuple[str, float]]:
    n = len(ranked)
    kept = []
    for i, (b, sc) in enumerate(ranked):
        pb = percentile_bin(i, n, bin_size=bin_size)
        if pb in bins:
            kept.append((b, sc))
    return kept


# -------------------------
# Loser List context + expression resolver (ported)
# -------------------------


LETTERS = list("ABCDEFGHIJ")


def heat_order(rows10: List[List[str]]) -> List[str]:
    c = Counter(d for r in rows10 for d in r)
    for d in DIGITS:
        c.setdefault(d, 0)
    return sorted(DIGITS, key=lambda d: (-c[d], d))


def rank_of_digit(order: List[str]) -> Dict[str, int]:
    return {d: i + 1 for i, d in enumerate(order)}


def neighbors(letter: str, span: int = 1) -> List[str]:
    i = LETTERS.index(letter)
    lo, hi = max(0, i - span), min(9, i + span)
    return LETTERS[lo:hi + 1]


def digits_for_letters_currentmap(letters: Set[str], digit_current_letters: Dict[str, str]) -> List[str]:
    return [d for d in DIGITS if digit_current_letters.get(d) in letters]


def compute_maps(last13: List[str]) -> Tuple[Dict, Dict]:
    rows = [list(s) for s in last13]
    prev10, curr10 = rows[1:11], rows[0:10]
    order_prev = heat_order(prev10)
    order_curr = heat_order(curr10)

    def pack(order, rows10):
        cnt = Counter(d for r in rows10 for d in r)
        for d in DIGITS:
            cnt.setdefault(d, 0)
        rank = rank_of_digit(order)
        d2L = {d: LETTERS[rank[d] - 1] for d in DIGITS}
        return dict(order=order, rank=rank, counts=cnt, digit_letters=d2L)

    return pack(order_prev, prev10), pack(order_curr, curr10)


def loser_list_ctx(last13_mr_to_oldest: List[str], last20_mr_to_oldest: Optional[List[str]] = None) -> Dict:
    if len(last13_mr_to_oldest) < 13:
        raise ValueError("Need at least 13 draws for Loser List context")

    rows = [list(s) for s in last13_mr_to_oldest]
    info_prev, info_curr = compute_maps(last13_mr_to_oldest)

    seed_digits_s = rows[0]
    prev_digits_s = rows[1]
    prev2_digits_s = rows[2] if len(rows) > 2 else []

    digit_prev_letters = info_prev["digit_letters"]
    digit_curr_letters = info_curr["digit_letters"]

    prev_core_letters = sorted({digit_prev_letters[d] for d in seed_digits_s}, key=lambda L: LETTERS.index(L))

    ring_letters = set()
    for L in prev_core_letters:
        ring_letters.update(neighbors(L, 1))
    ring_digits = digits_for_letters_currentmap(ring_letters, digit_curr_letters)

    loser_7_9 = info_curr["order"][-3:]
    curr_core_letters = sorted({digit_curr_letters[d] for d in seed_digits_s}, key=lambda L: LETTERS.index(L))
    new_core_digits = digits_for_letters_currentmap(set(curr_core_letters), digit_curr_letters)
    cooled_digits = [d for d in DIGITS if info_curr["rank"][d] > info_prev["rank"][d]]
    hot7_last10 = info_curr["order"][:7]

    prev_mirror_digits = sorted({str(MIRROR[int(d)]) for d in prev_digits_s}, key=int)
    union_last2 = sorted(set(prev_digits_s) | set(prev2_digits_s), key=int)
    due2 = sorted(set(DIGITS) - set(prev_digits_s) - set(prev2_digits_s), key=int)

    prev_core_currentmap_digits = digits_for_letters_currentmap(set(prev_core_letters), digit_curr_letters)
    edge_AC = digits_for_letters_currentmap(set("ABC"), digit_curr_letters)
    edge_HJ = digits_for_letters_currentmap(set("HIJ"), digit_curr_letters)

    seed_sum = sum(int(x) for x in seed_digits_s)
    prev_sum = sum(int(x) for x in prev_digits_s)

    core_size = len(prev_core_letters)
    core_size_flags = {
        "core_size_eq_2": core_size == 2,
        "core_size_eq_5": core_size == 5,
        "core_size_in_2_5": core_size in {2, 5},
        "core_size_in_235": core_size in {2, 3, 5},
    }

    seedp1 = sorted({str((int(d) + 1) % 10) for d in seed_digits_s}, key=int)
    counts_last2 = Counter(prev_digits_s + prev2_digits_s)
    for d in DIGITS:
        counts_last2.setdefault(d, 0)
    carry2_order = sorted(DIGITS, key=lambda d: (-counts_last2[d], int(d)))
    carry2 = carry2_order[:2]
    union2 = sorted(set(carry2) | set(seedp1), key=int)

    seed_pos = [seed_digits_s[i] for i in range(5)]
    p1_pos = [str((int(x) + 1) % 10) for x in seed_pos]
    union_digits = sorted(set(seed_pos) | set(p1_pos), key=int)

    ctx = dict(
        seed_digits=seed_digits_s,
        prev_digits=prev_digits_s,
        prev2_digits=prev2_digits_s,
        prev_mirror_digits=prev_mirror_digits,
        union_last2=union_last2,
        due_last2=due2,

        digit_prev_letters=digit_prev_letters,
        digit_current_letters=digit_curr_letters,
        prev_core_letters=prev_core_letters,
        curr_core_letters=curr_core_letters,
        prev_core_currentmap_digits=prev_core_currentmap_digits,

        ring_digits=ring_digits,
        new_core_digits=new_core_digits,
        cooled_digits=cooled_digits,
        loser_7_9=loser_7_9,
        hot7_last10=hot7_last10,
        edge_AC=edge_AC,
        edge_HJ=edge_HJ,
        seed_sum=seed_sum,
        prev_sum=prev_sum,
        core_size_flags=core_size_flags,
        seedp1=seedp1,
        seed_plus1=seedp1,
        seed_plus_1=seedp1,
        carry2=carry2,
        carry_top2=carry2,
        union2=union2,
        UNION2=union2,
        seed_pos=seed_pos,
        p1_pos=p1_pos,
        union_digits=union_digits,
        UNION_DIGITS=union_digits,
        current_map_order="".join(info_curr["order"]),
        previous_map_order="".join(info_prev["order"]),
        current_counts=info_curr["counts"],
        previous_counts=info_prev["counts"],
        current_rank=info_curr["rank"],
        previous_rank=info_prev["rank"],
    )

    # hot7_last20 if provided
    if last20_mr_to_oldest and len(last20_mr_to_oldest) >= 20:
        c = Counter(d for s in last20_mr_to_oldest[:20] for d in s)
        for d in DIGITS:
            c.setdefault(d, 0)
        hot7_20 = [d for d, _ in c.most_common(7)]
        ctx["hot7_last20"] = hot7_20
    else:
        ctx["hot7_last20"] = []

    # transitions used by some loserlist filters (placeholders in that csv)
    # We define them dynamically from letter transitions between prev_core and curr_core.
    # F→I means letters in ['F','G','H','I']
    fi_letters = set(["F", "G", "H", "I"])
    gi_letters = set(["G", "H", "I"])
    ctx["trans_FI"] = digits_for_letters_currentmap(fi_letters, digit_curr_letters)
    ctx["trans_GI"] = digits_for_letters_currentmap(gi_letters, digit_curr_letters)

    return ctx


def normalize_expr(expr: str) -> str:
    x = _normalize_quotes(expr or "").strip()
    x = x.replace('!==', '!=')
    x = x.replace('≤', '<=').replace('≥', '>=')
    return x


def resolve_loserlist_expression(expr: str, ctx: Dict) -> str:
    """Port of LoserList resolver: converts membership to int-safe sets."""
    x = normalize_expr(expr)

    list_vars = {
        "cooled_digits": ctx["cooled_digits"],
        "new_core_digits": ctx["new_core_digits"],
        "loser_7_9": ctx["loser_7_9"],
        "ring_digits": ctx["ring_digits"],
        "hot7_last10": ctx["hot7_last10"],
        "hot7_last20": ctx.get("hot7_last20", []),
        "seed_digits": ctx["seed_digits"],
        "prev_digits": ctx["prev_digits"],
        "prev_mirror_digits": ctx["prev_mirror_digits"],
        "union_last2": ctx["union_last2"],
        "due_last2": ctx["due_last2"],
        "prev_core_currentmap_digits": ctx["prev_core_currentmap_digits"],
        "edge_AC": ctx["edge_AC"],
        "edge_HJ": ctx["edge_HJ"],
        "trans_FI": ctx.get("trans_FI", []),
        "trans_GI": ctx.get("trans_GI", []),

        # positional shorthands
        "s1": [ctx["seed_pos"][0]], "S1": [ctx["seed_pos"][0]],
        "s2": [ctx["seed_pos"][1]], "S2": [ctx["seed_pos"][1]],
        "s3": [ctx["seed_pos"][2]], "S3": [ctx["seed_pos"][2]],
        "s4": [ctx["seed_pos"][3]], "S4": [ctx["seed_pos"][3]],
        "s5": [ctx["seed_pos"][4]], "S5": [ctx["seed_pos"][4]],

        "p1": ctx.get("seedp1", []), "P1": ctx.get("seedp1", []),
        "p2": [ctx["p1_pos"][1]], "P2": [ctx["p1_pos"][1]],
        "p3": [ctx["p1_pos"][2]], "P3": [ctx["p1_pos"][2]],
        "p4": [ctx["p1_pos"][3]], "P4": [ctx["p1_pos"][3]],
        "p5": [ctx["p1_pos"][4]], "P5": [ctx["p1_pos"][4]],
        "seedp1": ctx.get("seedp1", []), "SEEDP1": ctx.get("seedp1", []),

        "c1": ctx.get("carry2", []), "C1": ctx.get("carry2", []),
        "c2": ctx.get("carry2", []), "C2": ctx.get("carry2", []),
        "u1": ctx.get("union2", []), "U1": ctx.get("union2", []),
        "u2": ctx.get("union2", []), "U2": ctx.get("union2", []),
        "u3": ctx.get("union2", []), "U3": ctx.get("union2", []),
        "u4": ctx.get("union2", []), "U4": ctx.get("union2", []),
        "u5": ctx.get("union2", []), "U5": ctx.get("union2", []),
        "u6": ctx.get("union2", []), "U6": ctx.get("union2", []),
        "u7": ctx.get("union2", []), "U7": ctx.get("union2", []),
        "union2": ctx.get("union2", []), "UNION2": ctx.get("union2", []),
        "union_digits": ctx.get("union_digits", []),
        "UNION_DIGITS": ctx.get("UNION_DIGITS", []),
    }

    def set_lit(xs: List[str]) -> str:
        return "{" + ",".join(str(int(d)) for d in xs) + "}"

    def _flatten_name_list(inner: str) -> str:
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        flat, seen = [], set()
        for p in parts:
            if (len(p) >= 3) and (p[0] == p[-1]) and (p[0] in "'\""):
                val = p[1:-1].strip()
                if len(val) == 1 and val.isdigit() and val not in seen:
                    seen.add(val)
                    flat.append(val)
                continue
            if len(p) == 1 and p.isdigit():
                if p not in seen:
                    seen.add(p)
                    flat.append(p)
                continue
            if p in list_vars:
                for d in list_vars[p]:
                    if d not in seen:
                        seen.add(d)
                        flat.append(d)
                continue
        return set_lit(flat)

    # Replace tuple/bracket lists first
    x = re.sub(r"\bnot\s+in\s*\(([^)]+)\)", lambda m: " not in " + _flatten_name_list(m.group(1)), x)
    x = re.sub(r"\bin\s*\(([^)]+)\)", lambda m: " in " + _flatten_name_list(m.group(1)), x)
    x = re.sub(r"\bnot\s+in\s*\[([^\]]+)\]", lambda m: " not in " + _flatten_name_list(m.group(1)), x)
    x = re.sub(r"\bin\s*\[([^\]]+)\]", lambda m: " in " + _flatten_name_list(m.group(1)), x)

    for name, arr in list_vars.items():
        lit = set_lit(arr)
        x = re.sub(rf"\bin\s+{re.escape(name)}\b", " in " + lit, x)
        x = re.sub(rf"\bnot\s+in\s+{re.escape(name)}\b", " not in " + lit, x)
        x = re.sub(rf"\bin\s*\(\s*{re.escape(name)}\s*\)", " in " + lit, x)
        x = re.sub(rf"\bnot\s+in\s*\(\s*{re.escape(name)}\s*\)", " not in " + lit, x)

    # generator-preserving int(d) casting for membership
    x = re.sub(r"(\bif\s+)d(\s+(?:not\s+)?in\s+)", r"\1int(d)\2", x)
    x = re.sub(r"\bsum\(\s*d\s+in\s+", "sum(int(d) in ", x)

    # fix broken sums that lost generator
    def _ensure_gen(m):
        inner = m.group(1)
        return f"sum(int(d) in {inner} for d in combo_digits)"

    x = re.sub(r"\bsum\(\s*int\(d\)\s+(?:not\s+)?in\s*([^)]+)\)", _ensure_gen, x)
    x = re.sub(r"\bsum\(\s*d\s+(?:not\s+)?in\s*([^)]+)\)", _ensure_gen, x)
    x = re.sub(r"\bsum\(\s*1\s*for\s+in\s+combo_digits", "sum(1 for d in combo_digits", x)

    # letter membership literals
    def letter_contains(txt: str, varname: str, letters: Set[str]) -> str:
        p = re.compile(r"'([A-J])'\s+in\s+" + re.escape(varname))
        return p.sub(lambda mm: "True" if mm.group(1) in letters else "False", txt)

    x = letter_contains(x, "prev_core_letters", set(ctx["prev_core_letters"]))
    x = letter_contains(x, "core_letters", set(ctx["curr_core_letters"]))

    # scalars
    x = re.sub(r"\bseed_sum\b", str(ctx.get("seed_sum", 0)), x)
    x = re.sub(r"\bprev_sum\b", str(ctx.get("prev_sum", 0)), x)
    for key, val in (ctx.get("core_size_flags") or {}).items():
        x = re.sub(rf"\b{re.escape(key)}\b", "True" if val else "False", x)

    return x


# -------------------------
# Filter loading: LoserList CSV-like TXT + Batch10 CSV
# -------------------------


@dataclass(init=False)
class FilterDef:
    """Filter definition.

    Backwards-compatible with older filter CSV/TXT exports.
    Some files use column name `enabled`, while the app uses `enabled_default`.
    This constructor accepts either keyword to avoid Streamlit Cloud crashes.
    """

    fid: str
    name: str
    enabled_default: bool
    applicable_if: str
    expression: str
    source: str

    def __init__(
        self,
        fid: str,
        name: str,
        enabled_default: bool = True,
        applicable_if: str = "",
        expression: str = "",
        source: str = "",
        # Back-compat alias:
        enabled: bool | None = None,
        **_ignored: Any,
    ) -> None:
        if enabled is not None:
            enabled_default = enabled
        self.fid = fid
        self.name = name
        self.enabled_default = bool(enabled_default)
        self.applicable_if = applicable_if or ""
        self.expression = expression or ""
        self.source = source or ""


def _robust_read_filter_table(text: str, source: str) -> pd.DataFrame:
    """Parse filter CSV/TXT robustly.

    The Batch10 and LoserList exports sometimes contain commas inside expressions or minor
    column-count inconsistencies. Streamlit Cloud's pandas C-engine is strict and will
    raise tokenizing errors.

    Strategy:
    - Use Python's csv.reader (handles quoted commas).
    - For rows with too many columns: merge overflow into the last column.
    - For rows with too few columns: pad empties.
    - Ensure core columns exist: id/name/enabled/applicable_if/expression.
    """
    import csv

    f = io.StringIO(text)
    reader = csv.reader(f, delimiter=",", quotechar='"', escapechar='\\')
    rows = list(reader)
    if not rows:
        return pd.DataFrame(columns=["id", "name", "enabled", "applicable_if", "expression"])

    header = rows[0]

    # Normalize header length (prefer 15 like the python filter tester)
    target_cols = max(len(header), 15)
    if len(header) < target_cols:
        header = header + [f"Unnamed: {i}" for i in range(len(header), target_cols)]
    elif len(header) > target_cols:
        header = header[: target_cols - 1] + [",".join(header[target_cols - 1 :])]

    data = []
    for ridx, row in enumerate(rows[1:], start=2):
        if not row or all(str(x).strip() == "" for x in row):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header) - 1] + [",".join(row[len(header) - 1 :])]
        data.append(row[: len(header)])

    df = pd.DataFrame(data, columns=header).fillna("")

    # Normalize column names
    cols_lower = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)

    # If core columns are missing, assume first 5 columns map to core
    core = ["id", "name", "enabled", "applicable_if", "expression"]
    missing = [c for c in core if c not in df.columns]
    if missing:
        # rename first 5 columns to core, keep the rest
        cols = list(df.columns)
        for i, c in enumerate(core):
            if i < len(cols):
                cols[i] = c
        df.columns = cols

    # Ensure core columns exist
    for c in core:
        if c not in df.columns:
            df[c] = ""

    return df


def load_filter_csv_text(text: str, source: str) -> List[FilterDef]:
    """Load filters from CSV/TXT.

    Accepts:
    - The 15-column python-filter-tester CSV format
    - A minimal 5-column filter export

    Always returns FilterDef rows with id/name/enabled/applicable_if/expression.
    """
    df = _robust_read_filter_table(text, source)

    out: List[FilterDef] = []
    for _, r in df.iterrows():
        fid = str(r.get("id", "")).strip()
        if not fid or fid.lower() == "id":
            continue
        name = str(r.get("name", "")).strip() or fid
        enabled_raw = str(r.get("enabled", "TRUE")).strip().upper()
        enabled = enabled_raw in ("TRUE", "1", "YES", "Y")
        applicable_if = str(r.get("applicable_if", "")).strip()
        expr = str(r.get("expression", "")).strip()
        if not expr:
            continue
        out.append(FilterDef(fid=fid, name=name, enabled_default=enabled, applicable_if=applicable_if, expression=expr, source=source))

    return out

def dedupe_filters(filters: List[FilterDef]) -> List[FilterDef]:
    """Remove redundant filters by normalized expression + applicable_if."""
    seen = {}
    out = []
    for f in filters:
        key = (re.sub(r"\s+", " ", f.applicable_if.strip()), re.sub(r"\s+", " ", f.expression.strip()))
        if key in seen:
            continue
        seen[key] = f.fid
        out.append(f)
    return out



from dataclasses import dataclass

@dataclass
class FailedFilterEval:
    id: str
    name: str
    expression: str
    error_type: str
    error_message: str
    missing_vars: str


class FailureTracker:
    def __init__(self) -> None:
        self.items: list[FailedFilterEval] = []

    def has_any(self) -> bool:
        """Return True if any filter evaluation failures were recorded."""
        return bool(self.items)

    def add(self, *, fid: str, name: str, expression: str, err: Exception) -> None:
        missing = ""
        try:
            msg = str(err)
            m = re.search(r"name '([^']+)' is not defined", msg)
            if m:
                missing = m.group(1)
        except Exception:
            missing = ""

        self.items.append(
            FailedFilterEval(
                id=fid,
                name=name,
                expression=expression,
                error_type=type(err).__name__,
                error_message=str(err),
                missing_vars=missing,
            )
        )

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([
            {
                "id": x.id,
                "name": x.name,
                "expression": x.expression,
                "error_type": x.error_type,
                "error_message": x.error_message,
                "missing_vars": x.missing_vars,
            }
            for x in self.items
        ])



def sanitize_expr(expr: str) -> str:
    """Fix a couple common CSV-expression typos that trigger Python SyntaxWarning spam.

    Example: 'combo_sum <13and ...'  -> 'combo_sum <13 and ...'
    """
    if expr is None:
        return ""
    s = str(expr)
    # Strip stray tag tokens like '0PWK' that can leak into the expression field
    s = re.sub(r"\b\d+PWK\b", "", s)
    # Insert missing space between a number and a boolean keyword (and/or/not/in/is)
    s = re.sub(r"(\d)(and|or|not|in|is)\b", r"\1 \2", s)
    return s


_COMPILED_EXPR_CACHE = {}  # expr string -> compiled code


def safe_eval(expr: str, env: dict, *, filt: "FilterDef" | None = None, tracker: "FailureTracker" | None = None) -> bool:
    """Safely evaluate a boolean expression.

    - Uses a tiny compiled-expression cache for speed.
    - Keeps the failsafe ON: on any exception, returns False.
    - If a FailureTracker is provided, it records the failure so you can fix the filter.
    """
    expr = sanitize_expr(expr)
    try:
        code = _COMPILED_EXPR_CACHE.get(expr)
        if code is None:
            code = compile(expr, '<filter_expr>', 'eval')
            _COMPILED_EXPR_CACHE[expr] = code
        return bool(eval(code, {"__builtins__": {}}, env))
    except Exception as e:
        if tracker is not None and filt is not None:
            try:
                tracker.add(fid=filt.fid, name=filt.name, expression=expr, err=e)
            except Exception:
                pass
        return False
def build_filter_env(seed: str, prev: str, prev2: str, combo_box: str, extra_ctx: Optional[Dict] = None) -> Dict:
    seed_digits = [int(d) for d in seed]
    prev_digits = [int(d) for d in prev]
    prev2_digits = [int(d) for d in prev2]
    combo_digits = [int(d) for d in combo_box]

    # hot/cold/due (basic)
    hot, cold = hot_cold_sets([seed, prev, prev2], hot_k=3, cold_k=3)
    due = due_last2(prev, prev2)

    seed_counts = Counter(seed_digits)
    combo_sum = sum(combo_digits)
    prev_pattern = tuple([sum_category(sum(prev2_digits)), "Even" if sum(prev2_digits) % 2 == 0 else "Odd",
                          sum_category(sum(prev_digits)), "Even" if sum(prev_digits) % 2 == 0 else "Odd",
                          sum_category(sum(seed_digits)), "Even" if sum(seed_digits) % 2 == 0 else "Odd"])

    env = {
        "seed_value": int(seed),
        "seed_sum": sum(seed_digits),
        "prev_seed_sum": sum(prev_digits),
        "prev_prev_seed_sum": sum(prev2_digits),
        "seed_digits": seed_digits,
        "prev_seed_digits": prev_digits,
        "prev_prev_seed_digits": prev2_digits,
        "prev_pattern": prev_pattern,
        "hot_digits": hot,
        "cold_digits": cold,
        "due_digits": due,
        "seed_counts": seed_counts,
        "combo_digits": combo_digits,
        "combo_sum": combo_sum,
        "combo_sum_cat": sum_category(combo_sum),
        "seed_vtracs": set(V_TRAC_GROUPS[d] for d in seed_digits),
        "combo_vtracs": set(V_TRAC_GROUPS[d] for d in combo_digits),
        "common_to_both": set(seed_digits) & set(prev_digits),
        "mirror": MIRROR,
        "MIRROR": MIRROR,
        "V_TRAC": V_TRAC_GROUPS,
        "V_TRAC_GROUPS": V_TRAC_GROUPS,
        "vtrac": V_TRAC_GROUPS,
    }
    if extra_ctx:
        env.update(extra_ctx)
    return env


def evaluate_filters_on_pool(
    pool: List[str],
    filters: List[FilterDef],
    seed: str,
    prev: str,
    prev2: str,
    loser_ctx: Optional[Dict] = None,
) -> Tuple[List[str], List[Tuple[str, int, int]], FailureTracker]:
    """Apply enabled filters in order.

    Returns:
      - survivors: list of box strings
      - log: (filter_id, before_count, after_count)
      - failure_tracker: captures any filter eval exceptions (failsafe stays ON)
    """
    survivors = pool
    log: List[Tuple[str, int, int]] = []
    ft = FailureTracker()

    for f in filters:
        before = len(survivors)
        if before == 0:
            break
        if not f.enabled_default:
            continue

        # Resolve expressions
        app_if = f.applicable_if or "True"
        expr = f.expression or "False"
        if f.source == "loserlist":
            if loser_ctx is None:
                # cannot apply LL filters without ctx
                continue
            app_if = resolve_loserlist_expression(app_if, loser_ctx)
            expr = resolve_loserlist_expression(expr, loser_ctx)

        kept: List[str] = []

        for box in survivors:
            extra = loser_ctx if (f.source == "loserlist" and loser_ctx is not None) else {}
            env = build_filter_env(seed, prev, prev2, box, extra_ctx=extra)

            # applicable_if: if it errors, treat as NOT applicable (failsafe) but record it
            if not safe_eval(app_if, env, filt=f, tracker=ft):
                kept.append(box)
                continue

            # library meaning: expression is ELIMINATE-if
            if safe_eval(expr, env, filt=f, tracker=ft):
                # eliminated
                pass
            else:
                kept.append(box)

        survivors = kept
        after = len(survivors)
        log.append((f.fid, before, after))

    return survivors, log, ft



# -------------------------
# Walk-forward calibration
# -------------------------


@dataclass
class CalibrationResult:
    learning_window: int
    validation_window: int
    topN: int
    bins: Set[int]
    bin_size: int
    retain_rate: float
    avg_rank_of_winner: float
    details: pd.DataFrame


def build_transitions(df_mr: pd.DataFrame) -> List[Tuple[str, str]]:
    # df_mr is MR -> Oldest; transition is (seed=draw_i, next=draw_{i-1})? Actually seed is previous draw
    # We define seed as draw at index i (more recent), winner as index i-1 (older)??
    # For seed->next winner in chronological forward time, we use (older_seed -> newer_winner).
    # Since df is MR->Oldest, chronological forward is reverse. So:
    # oldest ... newer ... most recent
    results = df_mr["result"].tolist()
    chron = list(reversed(results))  # Oldest -> Most recent
    out = []
    for i in range(len(chron) - 1):
        out.append((chron[i], chron[i + 1]))
    return out


def calibrate_windows(
    df_mr: pd.DataFrame,
    candidate_topN: int,
    learn_options: List[int],
    val_options: List[int],
    gen_mode: str,
    gen_cap: int,
    bin_size: int = 5,
) -> CalibrationResult:
    transitions = build_transitions(df_mr)
    n_trans = len(transitions)
    if n_trans < 150:
        raise ValueError(f"Not enough history transitions for calibration (need ~150+, have {n_trans}).")

    best: Optional[CalibrationResult] = None

    for lw in learn_options:
        for vw in val_options:
            if lw + vw + 5 > n_trans:
                continue

            # Use the most recent chunk for validation
            val_start = n_trans - vw
            learn_start = max(0, val_start - lw)
            learn_trans = transitions[learn_start:val_start]
            val_trans = transitions[val_start:]

            # Recent draws for hot/cold context: last 20 winners in MR space
            recent_draws = df_mr.iloc[:20]["result"].tolist()
            tend = learn_tendencies_from_transitions(learn_trans, recent_draws)

            ranked_lists = []
            winners = []
            ranks = []
            kept_counts = []

            # Build ranked list for each validation transition
            for seed, winner in val_trans:
                boxes = generate_candidate_boxes(seed, mode=gen_mode, cap=gen_cap)
                ranked_scored = rank_candidates(boxes, seed, tend)
                ranked_scored = ranked_scored[:candidate_topN]
                ranked = [b for b, _ in ranked_scored]

                ranked_lists.append(ranked)
                winners.append(winner)

                wbox = "".join(sorted(winner))
                if wbox in ranked:
                    ranks.append(ranked.index(wbox) + 1)
                else:
                    ranks.append(None)
                kept_counts.append(len(ranked))

            heavy_bins = compute_winner_heavy_bins(ranked_lists, winners, bin_size=bin_size, keep_adjacent=True)

            hits = 0
            used = 0
            after_bins_ranks = []
            for ranked, winner in zip(ranked_lists, winners):
                if not ranked:
                    continue
                used += 1
                # simulate percentile bin trimming
                n = len(ranked)
                kept = []
                for i, b in enumerate(ranked):
                    pb = percentile_bin(i, n, bin_size=bin_size)
                    if pb in heavy_bins:
                        kept.append(b)
                wbox = "".join(sorted(winner))
                if wbox in kept:
                    hits += 1
                    after_bins_ranks.append(kept.index(wbox) + 1)

            retain_rate = hits / max(1, used)
            avg_rank = float(sum(after_bins_ranks) / len(after_bins_ranks)) if after_bins_ranks else float("inf")

            details = pd.DataFrame({
                "seed": [s for s, _ in val_trans],
                "winner": [w for _, w in val_trans],
                "winner_in_topN": [r is not None for r in ranks],
                "rank_in_topN": ranks,
            })

            cand = CalibrationResult(
                learning_window=lw,
                validation_window=vw,
                topN=candidate_topN,
                bins=heavy_bins,
                bin_size=bin_size,
                retain_rate=retain_rate,
                avg_rank_of_winner=avg_rank,
                details=details,
            )

            if best is None:
                best = cand
            else:
                # Priority: higher retain_rate; tie-break: lower avg winner rank
                if (cand.retain_rate > best.retain_rate) or (
                    abs(cand.retain_rate - best.retain_rate) < 1e-9 and cand.avg_rank_of_winner < best.avg_rank_of_winner
                ):
                    best = cand

    if best is None:
        raise ValueError("Could not calibrate with the provided history (try more history or smaller windows).")
    return best


# -------------------------
# Filter ranking / selection (dynamic)
# -------------------------


@dataclass
class FilterPerf:
    fid: str
    name: str
    source: str
    apply_count: int
    keep_rate: float
    avg_elim_rate: float





# -------------------------
# Filter expression helpers (fast ranking)
# -------------------------


def resolve_expr(expr: str, env: dict, *, source: str = 'batch10') -> str:
    """Resolve/normalize an expression for evaluation.

    - Normalizes unicode operators (≤/≥), smart quotes, etc.
    - Strips stray tokens like '0PWK' that sometimes leak into expressions.
    - For LoserList expressions, expands list variables into int-safe sets using the
      LoserList resolver with the current LL context (env['ll_ctx']).
    """
    if not expr:
        return 'False'
    x = normalize_expr(expr)
    x = sanitize_expr(x)
    if source == 'loserlist':
        try:
            ll_ctx = env.get('ll_ctx') or {}
            return resolve_loserlist_expression(x, ll_ctx)
        except Exception:
            return x
    return x



def make_env(seed: str, combo_or_winner: str, ll_ctx: dict | None = None) -> dict:
    """Build the eval environment for a given seed and combo.

    Notes:
    - Filters in this app are treated as BOX-based.
      So we convert any 5-digit string into its sorted (box) form.
    - If ll_ctx exists, we extract prev/prev2 from it so due/hot/cold logic matches.
    - We also attach ll_ctx onto the env so resolve_expr() can expand LoserList lists.
    """
    s = str(seed).zfill(5)
    digits = [d for d in str(combo_or_winner) if d.isdigit()]
    combo = ''.join(sorted(digits)) if len(digits) == 5 else str(combo_or_winner)

    if ll_ctx is not None:
        try:
            prev = ''.join(ll_ctx.get('prev_digits', [])) or s
        except Exception:
            prev = s
        try:
            prev2 = ''.join(ll_ctx.get('prev2_digits', [])) or prev
        except Exception:
            prev2 = prev
        extra = ll_ctx
    else:
        prev = s
        prev2 = s
        extra = {}

    env = build_filter_env(s, prev, prev2, combo, extra_ctx=extra)
    env['ll_ctx'] = ll_ctx or {}
    return env

def rank_filters_walkforward(
    df_mr: pd.DataFrame,
    filters: list[FilterDef],
    cal: CalibrationResult,
    gen_mode: str,
    gen_cap: int,
    target_after_bins: int,
    safety_keep_rate: float,
    sample_size: int = 80,
    max_filters: int | None = None,
    max_seconds: int = 90,
) -> list[FilterPerf]:
    """Rank filters using walk-forward winner retention, but fast.

    Key speedups vs the old version:
    - Winner retention is computed by evaluating filters ONLY on the real next-winner.
    - Elimination rate is estimated via a sample of the candidate pool (sample_size), not the full pool.
    - Expressions/applicable_if strings are still resolved per-transition, but we compile-cache eval.
    """
    import random, time

    t0 = time.time()
    rng = random.Random(1337)

    # Walk-forward transitions are in MR order: seed = df[i], winner = df[i-1]
    transitions = []  # (seed, winner, pool_boxes, ll_ctx)
    n = len(df_mr)

    learn = max(0, min(cal.learning_window, n - 2))
    validate = max(0, min(cal.validation_window, n - 2 - learn))
    total = learn + validate
    if total <= 0:
        return []



    # Tendencies used for ranking candidates (same approach as main run)
    try:
        trans_all = build_transitions(df_mr)
        n_trans = len(trans_all)
        vw = int(cal.validation_window)
        lw = int(cal.learning_window)
        val_start = n_trans - vw
        learn_start = max(0, val_start - lw)
        learn_trans = trans_all[learn_start:val_start]
        recent_draws = df_mr.iloc[:20]["result"].tolist()
        tend = learn_tendencies_from_transitions(learn_trans, recent_draws)
    except Exception:
        # extremely defensive fallback
        tend = learn_tendencies_from_transitions([], df_mr.iloc[:20]["result"].tolist() if len(df_mr) else [])
    # Build transitions
    for idx in range(1, total + 1):
        seed = str(df_mr.iloc[idx]["result"]).zfill(5)
        winner = str(df_mr.iloc[idx - 1]["result"]).zfill(5)

        # pool generation + ranking + percentile keep (same logic as main run)
        boxes = generate_candidate_boxes(seed, mode=gen_mode, cap=int(gen_cap))
        ranked = rank_candidates(boxes, seed, tend)
        ranked = ranked[: cal.topN]
        ranked_bins = apply_percentile_bins(ranked, cal.bins, bin_size=cal.bin_size)
        pool = [b for b, _ in ranked_bins]
        pool = pool[: max(int(target_after_bins), 1)]

        # build LL ctx for this seed window (cheap)
        sub13 = df_mr.iloc[idx: idx + 13]["result"].tolist()
        sub20 = df_mr.iloc[idx: idx + 20]["result"].tolist()
        try:
            ll = loser_list_ctx(sub13, sub20)
        except Exception:
            ll = None

        transitions.append((seed, winner, pool, ll))

    # Evaluate filters
    perfs: list[FilterPerf] = []

    # optionally cap number of filters evaluated (for quick runs)
    filters_iter = filters
    if max_filters is not None:
        filters_iter = filters_iter[: max(0, int(max_filters))]

    for f in filters_iter:
        if (time.time() - t0) > max_seconds:
            break

        kept = 0
        applied = 0
        elim_rates = []

        for seed, winner, pool, ll in transitions:
            # Winner check (retention)
            w_env = make_env(seed, winner, ll)

            app_if = (f.applicable_if or '').strip()
            if app_if:
                try:
                    app_ok = safe_eval(resolve_expr(app_if, w_env, source=f.source), w_env)
                except Exception:
                    app_ok = False
            else:
                app_ok = True

            if not app_ok:
                # not applicable -> treated as kept and not counted in applied
                kept += 1
                continue

            applied += 1

            # If filter returns True => eliminate
            try:
                elim_w = safe_eval(resolve_expr(f.expression, w_env, source=f.source), w_env)
            except Exception:
                elim_w = False

            if not elim_w:
                kept += 1

            # Estimate elim rate using a sample of pool
            if pool:
                k = min(int(sample_size), len(pool))
                sample = pool if k == len(pool) else rng.sample(pool, k)

                elim_ct = 0
                seen_ct = 0
                for box in sample:
                    env = make_env(seed, box, ll)

                    if app_if:
                        try:
                            ok = safe_eval(resolve_expr(app_if, env, source=f.source), env)
                        except Exception:
                            ok = False
                        if not ok:
                            continue

                    seen_ct += 1
                    try:
                        if safe_eval(resolve_expr(f.expression, env, source=f.source), env):
                            elim_ct += 1
                    except Exception:
                        continue

                if seen_ct:
                    elim_rates.append(elim_ct / seen_ct)

        if total == 0:
            continue

        keep_rate = kept / total
        avg_elim = (sum(elim_rates) / len(elim_rates)) if elim_rates else 0.0

        if keep_rate >= float(safety_keep_rate):
            perfs.append(
                FilterPerf(
                    fid=f.fid,
                    name=f.name,
                    keep_rate=keep_rate,
                    avg_elim_rate=avg_elim,
                    apply_count=applied,
                    source=f.source,
                )
            )

    # Sort: highest keep rate first, then higher elimination
    perfs.sort(key=lambda p: (-p.keep_rate, -p.avg_elim_rate, -p.apply_count, p.fid))
    return perfs

def straight_rankings_from_box(box: str, tend: LearnedTendencies, seed: str) -> List[str]:
    """Return a ranked list of straight permutations from a box.

    Heuristic:
      - Order digits by: (is_seed_digit desc, is_hot desc, count desc, digit asc)
      - Then generate unique permutations and score each by positional hotness.
    """
    digits = [int(x) for x in box]
    counts = Counter(digits)
    seed_set = set(int(x) for x in seed)
    hot_set = set(tend.hot_digits)

    # generate unique permutations
    perms = set(product(digits, repeat=5))
    # That explodes; instead generate permutations via backtracking respecting counts
    uniques = []

    def backtrack(path, remaining: Counter):
        if len(path) == 5:
            uniques.append(tuple(path))
            return
        for d in sorted(remaining.keys()):
            if remaining[d] <= 0:
                continue
            remaining[d] -= 1
            path.append(d)
            backtrack(path, remaining)
            path.pop()
            remaining[d] += 1

    backtrack([], Counter(digits))

    # score straights
    scored = []
    for tup in uniques:
        sc = 0.0
        for i, d in enumerate(tup):
            sc += 0.4 if d in seed_set else 0.0
            sc += 0.25 if d in hot_set else 0.0
            sc += 0.05 * counts[d]
            sc += 0.01 * (9 - abs(4 - i))  # mild center preference
        scored.append(("".join(str(x) for x in tup), sc))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [s for s, _ in scored]


# -------------------------
# UI
# -------------------------


st.title("Pick 5 — SeedGrid + Walk-Forward Auto Calibration + Dynamic Filters")

with st.sidebar:
    st.header("History")
    up = st.file_uploader("Upload Pick 5 history (TXT)", type=["txt", "csv"])
    stream_name = st.text_input("Stream label (for saving defaults)", value="STATE_STREAM")

    st.divider()
    st.header("Auto settings")
    topN_default = st.slider("Initial Top-N after ranking", min_value=200, max_value=2000, value=500, step=50)
    target_final = st.slider("Target final pool size", min_value=25, max_value=800, value=150, step=25)
    gen_cap = st.slider("Generation cap (box candidates)", min_value=2000, max_value=60000, value=20000, step=1000)
    gen_mode = st.selectbox(
        "Generation mode",
        ["two_pairs_plus_non_grid", "two_pairs_plus_grid", "mirror_pair_mix"],
        index=0,
        help="Controls how candidate boxes are generated BEFORE ranking/percentiles."
    )

    st.divider()
    st.header("Calibration windows")
    learn_opts = st.multiselect("Learning window candidates (transitions)", [120, 180, 240], default=[120, 180, 240])
    val_opts = st.multiselect("Validation window candidates (transitions)", [60, 90, 120], default=[90])
    bin_size = st.select_slider("Percentile bin size", options=[5, 10], value=5)

    st.divider()
    st.header("Filter sources")
    use_loser = st.checkbox("Include LoserList filters", value=True)
    loserlist_uploader = st.file_uploader("Upload LoserList filters (CSV/TXT)", type=["txt","csv"], help="Use the exported loserlist filters file (CSV-format with id,name,expression...).")
    use_batch10 = st.checkbox("Include Batch10 filters (first 780 rows)", value=True)
    batch10_uploader = st.file_uploader("Upload Batch10 filters CSV", type=["csv"], help="Upload lottery_filters_batch10.csv (we only use rows 1-780 per your instruction).")
    safety_keep_rate = st.slider("Min winner-keep rate to auto-apply filter", 0.50, 1.00, 0.75, 0.01)


if not up:
    st.info("Upload a Pick-5 history TXT to begin.")
    st.stop()


history_text = up.read().decode("utf-8", errors="ignore")

try:
    df = load_history_from_text(history_text)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Loaded {len(df)} results.")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.subheader("Most recent results")
    st.dataframe(df[["date", "result"]].head(10), use_container_width=True)
with colB:
    # Choose the seed by the most recent *parsed date* (not by current row order).
    if 'date' in df.columns and df['date'].notna().any():
        seed_idx = df['date'].idxmax()
    else:
        seed_idx = df.index[0]
    most_recent = str(df.loc[seed_idx, 'result']).zfill(5)
    seed_date = df.loc[seed_idx, 'date'] if 'date' in df.columns else None
    st.subheader('Seed (most recent)')
    st.code(most_recent)
    if seed_date is not None and str(seed_date) != 'NaT':
        st.caption(f'Seed date used: {seed_date}')
    # Optional manual override (helps debug when the input file has mixed orders).
    if st.checkbox('Manually choose seed row', value=False, key='manual_seed_toggle'):
        options = df[['date','result']].copy()
        options['label'] = options.apply(lambda r: f"{r['date']}  {str(r['result']).zfill(5)}", axis=1)
        chosen = st.selectbox('Select seed row', options['label'].tolist(), index=int(options.index.get_loc(seed_idx)), key='manual_seed_row')
        chosen_idx = options.index[options['label'] == chosen][0]
        most_recent = str(df.loc[chosen_idx, 'result']).zfill(5)
        seed_date = df.loc[chosen_idx, 'date'] if 'date' in df.columns else None
        st.info(f'Using manual seed: {most_recent} (date: {seed_date})')
with colC:
    st.subheader("Grid digits")
    g = make_grid_sets(most_recent)
    st.write({k: "".join(str(x) for x in v) for k, v in g.items()})



# Load filters (prefer sidebar uploads; fall back to bundled files if present)
# - This avoids hardcoded /mnt/data paths that do NOT exist on Streamlit Cloud.
# - It also makes the repo work out-of-the-box if the CSVs are committed.

def _read_text_if_exists(paths):
    for pp in paths:
        try:
            pth = pathlib.Path(pp)
            if pth.is_file():
                return pth.read_text(encoding='utf-8')
        except Exception:
            pass
    return None

filters_all = []

# --- LoserList filters ---
loser_text = None
if use_loser:
    if loserlist_uploader is not None:
        try:
            loser_text = loserlist_uploader.getvalue().decode('utf-8', errors='replace')
        except Exception as e:
            st.warning(f"Could not read uploaded LoserList filters: {e}")
            loser_text = None
    else:
        # Try common filenames in repo
        loser_text = _read_text_if_exists([
            'loserlist_filters15_FIXED.csv',
            'loserlist_filters15_FIXED (5).csv',
            'loserlist_filters15_FIXED(5).csv',
            'loserlist_filters15.csv',
            'loserlist_filters15.txt',
        ])

    if loser_text:
        try:
            ll_filters = load_filter_csv_text(loser_text, source='loserlist')
            for f in ll_filters:
                filters_all.append(f)
        except Exception as e:
            st.warning(f"Could not load LoserList filters: {e}")

# --- Batch10 filters ---
batch_text = None
if use_batch10:
    if batch10_uploader is not None:
        try:
            batch_text = batch10_uploader.getvalue().decode('utf-8', errors='replace')
        except Exception as e:
            st.warning(f"Could not read uploaded Batch10 filters: {e}")
            batch_text = None
    else:
        batch_text = _read_text_if_exists([
            'lottery_filters_batch10_NO_LOSERLIST.csv',
            'lottery_filters_batch10.csv',
            'lottery_filters_batch10_NO_LOSERLIST (1).csv',
        ])

    if batch_text:
        try:
            b10_filters = load_filter_csv_text(batch_text, source='batch10')
            # per instruction: use only first 780 rows
            b10_filters = b10_filters[:780]
            for f in b10_filters:
                filters_all.append(f)
        except Exception as e:
            st.warning(f"Could not load Batch10 filters: {e}")

# Dedupe by ID (prefer first occurrence)
_seen = set()
filters_dedup = []
for f in filters_all:
    fid = (f.fid or '').strip() or (f.name or '').strip()
    if fid in _seen:
        continue
    _seen.add(fid)
    filters_dedup.append(f)
filters_all = filters_dedup

st.caption(f"Filters loaded (after dedupe): {len(filters_all)}")


st.divider()
st.subheader("Auto Calibration (Walk-forward)")

# Auto-run calibration by default (so you don't have to remember steps).
# You can still re-run it if you change settings.
need_cal = "cal" not in st.session_state
rerun = st.button("Re-run auto calibration", type="primary", help="Recompute best windows + winner-heavy percentile zones.")
if need_cal or rerun:
    with st.spinner("Running auto calibration (walk-forward)..."):
        try:
            cal = calibrate_windows(
                df_mr=df,
                candidate_topN=int(topN_default),
                learn_options=sorted(set(learn_opts)) or [180],
                val_options=sorted(set(val_opts)) or [90],
                gen_mode=gen_mode,
                gen_cap=int(gen_cap),
                bin_size=int(bin_size),
            )
            st.session_state["cal"] = cal
        except Exception as e:
            st.error(str(e))
            st.stop()

cal: Optional[CalibrationResult] = st.session_state.get("cal")
st.success(
    f"Using calibration: learn={cal.learning_window}, validate={cal.validation_window}, "
    f"retain={cal.retain_rate:.3f}, avg rank={cal.avg_rank_of_winner:.1f}"
)
st.write("Winner-heavy percentile bins kept:", sorted(cal.bins))
with st.expander("Show recent calibration rows"):
    st.dataframe(cal.details.tail(30), use_container_width=True)


st.divider()
st.subheader("Run prediction for the current seed")

# Build learning tendencies using chosen learning window
transitions = build_transitions(df)
n_trans = len(transitions)
vw = cal.validation_window
lw = cal.learning_window
val_start = n_trans - vw
learn_start = max(0, val_start - lw)
learn_trans = transitions[learn_start:val_start]
recent_draws = df.iloc[:20]["result"].tolist()
tend = learn_tendencies_from_transitions(learn_trans, recent_draws)

seed = most_recent

boxes = generate_candidate_boxes(seed, mode=gen_mode, cap=int(gen_cap))
ranked_scored = rank_candidates(boxes, seed, tend)[:cal.topN]
ranked_scored_bins = apply_percentile_bins(ranked_scored, cal.bins, bin_size=cal.bin_size)

st.write(f"Generated boxes: {len(boxes)}")
st.write(f"Top-N after ranking: {len(ranked_scored)}")
st.write(f"After percentile-bin keep: {len(ranked_scored_bins)}")

ranked_pool = [b for b, _ in ranked_scored_bins]
ranked_pool = ranked_pool[: max(int(target_final) * 4, 450)]  # keep enough for filtering

st.caption(f"Pool entering dynamic filter ranking: {len(ranked_pool)}")


# Determine prev and prev2 from df (MR->Oldest)
prev = df.iloc[1]["result"] if len(df) > 1 else seed
prev2 = df.iloc[2]["result"] if len(df) > 2 else prev

# Build loser ctx for current run
last13 = df.iloc[:13]["result"].tolist()
last20 = df.iloc[:20]["result"].tolist()
ll_ctx = None
try:
    ll_ctx = loser_list_ctx(last13, last20)
except Exception:
    ll_ctx = None



st.subheader("Dynamic filter ranking (walk-forward)")

# Fast ranking controls
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    sample_size = st.number_input("Ranking sample size (boxes per transition)", min_value=10, max_value=400, value=80, step=10)
with c2:
    max_filters = st.number_input("Max filters to rank (0 = all)", min_value=0, max_value=2000, value=0, step=50)
with c3:
    max_seconds = st.number_input("Max seconds per rank run", min_value=15, max_value=600, value=90, step=15)

if "ranked_filters" not in st.session_state:
    st.session_state["ranked_filters"] = []

run_rank = st.button("Run / Refresh filter ranking", type="primary")

if run_rank:
    with st.spinner("Ranking filters (walk-forward, fast)..."):
        ranked_filters = rank_filters_walkforward(
            df_mr=df,
            filters=filters_all,
            cal=cal,
            gen_mode=gen_mode,
            gen_cap=int(gen_cap),
            target_after_bins=max(int(target_final) * 4, 450),
            safety_keep_rate=float(safety_keep_rate),
            sample_size=int(sample_size),
            max_filters=(None if int(max_filters) == 0 else int(max_filters)),
            max_seconds=int(max_seconds),
        )
        st.session_state["ranked_filters"] = ranked_filters

ranked_filters = st.session_state.get("ranked_filters", [])

records = [
    {
        "id": p.fid,
        "name": p.name,
        "source": p.source,
        "winner_keep_rate": round(p.keep_rate, 3),
        "avg_elim_rate": round(p.avg_elim_rate, 3),
        "apply_count": p.apply_count,
    }
    for p in ranked_filters
]

df_perf = pd.DataFrame.from_records(records)
if df_perf.empty:
    df_perf = pd.DataFrame(columns=["id", "name", "source", "winner_keep_rate", "avg_elim_rate", "apply_count"])
    st.info("No ranking computed yet. Click **Run / Refresh filter ranking**.")

st.dataframe(df_perf.head(50), use_container_width=True)
st.caption("Auto-apply order = safest → more aggressive (among those meeting keep-rate threshold)")



st.subheader("Apply filters automatically to reach target")

if len(ranked_filters) == 0:
    apply_count = 0
else:
    apply_count = st.slider("Max filters to auto-apply", 0, min(100, len(ranked_filters)), min(25, len(ranked_filters)))

    # Robustly pick the ID column (some runs may produce an empty df without columns)
    id_col = 'id' if 'id' in df_perf.columns else ('filter_id' if 'filter_id' in df_perf.columns else None)
    if id_col is None:
        to_apply_ids = set()
    else:
        to_apply_ids = set(df_perf.head(apply_count)[id_col].astype(str).tolist())
filters_to_apply = [f for f in filters_all if f.fid in to_apply_ids]

# Preserve the ranked order from df_perf
order_list = df_perf.head(apply_count)[id_col].astype(str).tolist() if id_col else []
order = {fid: i for i, fid in enumerate(order_list)}
filters_to_apply.sort(key=lambda f: order.get(f.fid, 999999))

survivors, log, failure_tracker = evaluate_filters_on_pool(
    pool=ranked_pool,
    filters=filters_to_apply,
    seed=seed,
    prev=prev,
    prev2=prev2,
    loser_ctx=ll_ctx,
)

st.write(f"After auto filters: **{len(survivors)}**")

log_df = pd.DataFrame(log, columns=["filter_id", "before", "after"])
st.dataframe(log_df, use_container_width=True)

# Show failed filters (failsafe kept, but nothing is silent)
if failure_tracker.has_any():
    st.warning(f"Filters failed (error → automatically returned False): {len(failure_tracker.items)}")
    with st.expander("Show failed filters (fix these in the filter files)", expanded=False):
        fail_df = failure_tracker.to_dataframe()
        st.dataframe(fail_df, use_container_width=True)

        st.download_button(
            "Download failed filters report (csv)",
            data=fail_df.to_csv(index=False),
            file_name=f"failed_filters_{stream_name}.csv",
        )

        # TXT summary (copy/paste-friendly)
        lines = []
        for row in failure_tracker.items:
            lines.append(f"{row.fid}: {row.name} → {row.error_type}: {row.message}")
            if row.missing_vars:
                lines.append(f"  missing_vars: {', '.join(row.missing_vars)}")
            lines.append(f"  expr: {row.expression}")
            lines.append("")
        st.download_button(
            "Download failed filters report (txt)",
            data="\n".join(lines).strip() + "\n",
            file_name=f"failed_filters_{stream_name}.txt",
        )


# Trim to target_final while keeping order by original ranking score
rank_index = {b: i for i, (b, _) in enumerate(ranked_scored_bins)}
survivors.sort(key=lambda b: rank_index.get(b, 10**9))
final_pool = survivors[:int(target_final)]

st.subheader("Final pool (box, ranked most-likely → least-likely)")

final_df = pd.DataFrame({
    "rank": list(range(1, len(final_pool) + 1)),
    "box": final_pool,
})
st.dataframe(final_df, use_container_width=True)


st.subheader("Straight recommendations (ranked)")

straight_limit = st.slider("How many box combos to expand into straights", 1, len(final_pool), min(25, len(final_pool)))
max_straights_per_box = st.slider("Max straights per box", 1, 120, 24)

straight_rows = []
for i, box in enumerate(final_pool[:straight_limit], start=1):
    straights = straight_rankings_from_box(box, tend, seed)[:max_straights_per_box]
    for j, s in enumerate(straights, start=1):
        straight_rows.append({"box_rank": i, "straight_rank": j, "straight": s, "box": box})

straight_df = pd.DataFrame(straight_rows)
st.dataframe(straight_df, use_container_width=True)


st.download_button(
    "Download final BOX list (txt)",
    data="\n".join(final_pool),
    file_name=f"pick5_final_box_{stream_name}.txt",
)

st.download_button(
    "Download straight recommendations (csv)",
    data=straight_df.to_csv(index=False),
    file_name=f"pick5_straights_{stream_name}.csv",
    mime="text/csv",
)
