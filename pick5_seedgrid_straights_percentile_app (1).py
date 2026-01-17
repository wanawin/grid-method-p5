import re
from collections import Counter, defaultdict
from itertools import product
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Pick 5 — Seed Grid + Straights
# -----------------------------

MIRROR = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}


def zfill_digits(s: str, n: int = 5) -> str:
    s = ''.join(ch for ch in (s or '') if ch.isdigit())
    return s.zfill(n)[-n:]


def parse_history_text(text: str) -> pd.DataFrame:
    """Parse a loose TXT history into a tidy DataFrame.

    Expected: one draw per line (any format), as long as the line contains a 5-digit result
    (with or without separators like 1-2-3-4-5).

    If the line contains a stream label (Midday/Evening/Night), it will be captured.
    """
    rows = []
    for raw_line in (text or '').splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # stream
        low = line.lower()
        stream = 'ALL'
        if 'midday' in low:
            stream = 'Midday'
        elif 'evening' in low:
            stream = 'Evening'
        elif 'night' in low:
            stream = 'Night'

        # date (best-effort)
        date_match = re.search(r'(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})', line)
        date = date_match.group(1) if date_match else ''

        # result: try separated digits first, then plain 5 digits
        m = re.search(r'(\d)[\s\-]?(\d)[\s\-]?(\d)[\s\-]?(\d)[\s\-]?(\d)', line)
        if not m:
            continue
        result = ''.join(m.groups())
        if len(result) != 5:
            continue

        rows.append({'date': date, 'stream': stream, 'result': result})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep chronological order if date parse works; otherwise keep input order.
    # We'll still treat the last row as the most recent.
    try:
        df['_dt'] = pd.to_datetime(df['date'], errors='coerce')
        if df['_dt'].notna().any():
            # stable sort: unknown dates keep relative order
            df['_idx'] = range(len(df))
            df = df.sort_values(by=['_dt', '_idx'], kind='mergesort')
        df = df.drop(columns=[c for c in ['_dt', '_idx'] if c in df.columns])
    except Exception:
        pass

    df = df.reset_index(drop=True)
    return df


def digits_of(s: str) -> List[int]:
    s = zfill_digits(s, 5)
    return [int(ch) for ch in s]


def build_seed_grid(seed: str, lookback_pos_freq: Dict[int, Counter], row_order: str = 'Seed, +1, -1, Mirror') -> List[List[int]]:
    """Return 5 columns, each with 4 digits."""
    seed = zfill_digits(seed, 5)
    sd = [int(ch) for ch in seed]

    def plus1(d: int) -> int:
        return (d + 1) % 10

    def minus1(d: int) -> int:
        return (d - 1) % 10

    cols: List[List[int]] = []
    for pos, d in enumerate(sd):
        seed_d = d
        p1 = plus1(d)
        m1 = minus1(d)
        mir = MIRROR[d]

        if row_order == 'Seed, +1, -1, Mirror':
            base = [seed_d, p1, m1, mir]
        elif row_order == 'Seed, Mirror, +1, -1':
            base = [seed_d, mir, p1, m1]
        else:
            base = [seed_d, p1, m1, mir]

        # de-dup while preserving order
        seen = set()
        uniq = []
        for x in base:
            if x not in seen:
                uniq.append(x)
                seen.add(x)

        # If we lost a row because of duplicates (e.g., seed=0, mirror=5 distinct so ok;
        # but seed=5 mirror=0 etc still ok; duplicates mostly from +1/-1 collisions), pad to 4.
        if len(uniq) < 4:
            freq = lookback_pos_freq.get(pos, Counter())
            for cand, _ in freq.most_common():
                if cand not in seen:
                    uniq.append(cand)
                    seen.add(cand)
                if len(uniq) == 4:
                    break
        # final safety
        while len(uniq) < 4:
            for cand in range(10):
                if cand not in seen:
                    uniq.append(cand)
                    seen.add(cand)
                if len(uniq) == 4:
                    break
        cols.append(uniq[:4])

    return cols


def compute_position_frequencies(df: pd.DataFrame, stream: str, lookback: int) -> Dict[int, Counter]:
    sdf = df if stream == 'ALL' else df[df['stream'] == stream]
    if sdf.empty:
        return {i: Counter() for i in range(5)}
    tail = sdf.tail(lookback)
    pos_freq = {i: Counter() for i in range(5)}
    for r in tail['result'].tolist():
        digs = digits_of(r)
        for i, d in enumerate(digs):
            pos_freq[i][d] += 1
    return pos_freq


def compute_overall_frequencies(df: pd.DataFrame, stream: str, lookback: int) -> Counter:
    sdf = df if stream == 'ALL' else df[df['stream'] == stream]
    if sdf.empty:
        return Counter()
    tail = sdf.tail(lookback)
    c = Counter()
    for r in tail['result'].tolist():
        c.update(digits_of(r))
    return c


def score_straight(straight: Tuple[int, int, int, int, int], seed_digits: List[int], pos_freq: Dict[int, Counter], overall_freq: Counter,
                  hot: set, cold: set) -> float:
    # Base: position likelihood (Laplace smoothing)
    base = 0.0
    draws = max(1, sum(pos_freq[0].values()))
    for i, d in enumerate(straight):
        base += (pos_freq[i][d] + 1) / (draws + 10)

    # Inclusion boosts (soft)
    seed_set = set(seed_digits)
    mirror_set = {MIRROR[d] for d in seed_digits}
    plus_set = {(d + 1) % 10 for d in seed_digits}
    minus_set = {(d - 1) % 10 for d in seed_digits}

    digs = list(straight)
    boost = 0.0
    boost += 0.35 * sum(1 for d in digs if d in seed_set)
    boost += 0.22 * sum(1 for d in digs if d in mirror_set)
    boost += 0.18 * sum(1 for d in digs if d in plus_set)
    boost += 0.18 * sum(1 for d in digs if d in minus_set)

    # Hot/cold as small nudges
    boost += 0.10 * sum(1 for d in digs if d in hot)
    boost += 0.08 * sum(1 for d in digs if d in cold)

    # Mild penalty for extreme parity patterns (soft, not elimination)
    evens = sum(1 for d in digs if d % 2 == 0)
    if evens in (0, 5):
        boost -= 0.15

    # Mild penalty for all high/all low
    highs = sum(1 for d in digs if d >= 5)
    if highs in (0, 5):
        boost -= 0.10

    # Another tiny nudge: overall frequency of digits
    tot = max(1, sum(overall_freq.values()))
    boost += 0.05 * sum((overall_freq[d] + 1) / (tot + 10) for d in digs)

    return base + boost


def rank_boxes_from_grid(cols: List[List[int]], seed_digits: List[int], pos_freq: Dict[int, Counter], overall_freq: Counter,
                         hot: set, cold: set) -> pd.DataFrame:
    straights = list(product(*cols))

    best_by_box: Dict[str, Tuple[float, Tuple[int, int, int, int, int]]] = {}
    for s in straights:
        score = score_straight(s, seed_digits, pos_freq, overall_freq, hot, cold)
        box_key = ''.join(sorted(str(d) for d in s))
        prev = best_by_box.get(box_key)
        if prev is None or score > prev[0]:
            best_by_box[box_key] = (score, s)

    rows = []
    for box_key, (score, s) in best_by_box.items():
        rows.append({
            'box': box_key,
            'best_straight': ''.join(str(d) for d in s),
            'score': float(score),
        })

    out = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    n = len(out)
    if n > 1:
        out['rank'] = out.index + 1
        out['rank_pct'] = (out['rank'] - 1) / (n - 1) * 100
    else:
        out['rank'] = 1
        out['rank_pct'] = 0.0
    return out


def main():
    st.set_page_config(page_title='Pick 5 Seed Grid — Straights + Percentile', layout='wide')

    st.title('Pick 5 Seed Grid — Straights + Percentile Ranking')
    st.caption('Seed grid uses Seed / +1 / -1 / Mirror per position. Ranking uses inclusion boosts + recent-position likelihood. Trim using a single Top-N slider.')

    with st.sidebar:
        st.header('Inputs')
        up = st.file_uploader('Upload Pick-5 history (TXT/CSV-like)', type=['txt', 'csv'])
        stream = st.selectbox('Stream', ['ALL', 'Midday', 'Evening', 'Night'], index=0, help='Use the same stream you plan to play. If your file has no stream labels, use ALL.')
        lookback = st.slider('Lookback draws (learn stats)', min_value=20, max_value=365, value=120, step=5,
                             help='How many most-recent draws to learn digit tendencies from (same stream).')
        row_order = st.selectbox('Grid row order (display only)', ['Seed, +1, -1, Mirror', 'Seed, Mirror, +1, -1'],
                                 help='This only affects how the 4 digits are listed in each column. Ranking does NOT depend on row order.')

        st.divider()
        st.header('Output size')
        keep_n = st.slider('Keep Top-N box combos', min_value=10, max_value=400, value=60, step=5,
                           help='This is your only “tight vs wide” control. Increase if the winner keeps dropping in backtests.')

        st.divider()
        st.header('Debug')
        debug_show = st.checkbox('Show parsing + seed debug', value=False)

    if not up:
        st.info('Upload a Pick-5 history file to begin.')
        return

    raw = up.read().decode('utf-8', errors='ignore')
    df = parse_history_text(raw)
    if df.empty:
        st.error('No 5-digit results found in that file. Upload a file that contains Pick-5 results (e.g., 0-1-2-3-4 or 01234).')
        return

    sdf = df if stream == 'ALL' else df[df['stream'] == stream]
    if sdf.empty:
        st.error(f'No rows found for stream: {stream}. Try ALL, or check the file format.')
        return

    seed = sdf.iloc[-1]['result']  # most recent in that stream
    seed_digits = digits_of(seed)

    pos_freq = compute_position_frequencies(df, stream, lookback)
    overall_freq = compute_overall_frequencies(df, stream, lookback)

    # Hot/cold sets from overall frequency in lookback
    full_counts = {d: overall_freq.get(d, 0) for d in range(10)}
    hot = {d for d, _ in sorted(full_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]}
    cold = {d for d, _ in sorted(full_counts.items(), key=lambda kv: (kv[1], kv[0]))[:3]}

    cols = build_seed_grid(seed, pos_freq, row_order=row_order)

    if debug_show:
        st.subheader('Debug')
        st.write({
            'seed': seed,
            'seed_digits': seed_digits,
            'seed_mirror': [MIRROR[d] for d in seed_digits],
            'hot': sorted(hot),
            'cold': sorted(cold),
        })

    # Grid display
    st.subheader('Seed Grid (5 positions × 4 digits)')
    grid_df = pd.DataFrame({
        f'Pos {i+1}': col for i, col in enumerate(cols)
    }, index=['A', 'B', 'C', 'D'])
    st.dataframe(grid_df, use_container_width=True)

    # Rank boxes
    ranked = rank_boxes_from_grid(cols, seed_digits, pos_freq, overall_freq, hot, cold)

    # Top N
    topn = ranked.head(int(keep_n)).copy()

    # Winner check: if your file includes the next draw (i.e., if you're backtesting), show it
    # For now, we only show the most recent seed. You can paste a "target winner" to check.
    st.subheader('Ranked Box Combos (Top-N)')
    c1, c2 = st.columns([2, 1])
    with c2:
        check = st.text_input('Check a specific 5-digit result', help='Enter the actual 5-digit winner (e.g., 27500) to see if its BOX is in Top-N.')

    checked_msg = None
    if check:
        check = zfill_digits(check, 5)
        check_box = ''.join(sorted(check))
        found = topn[topn['box'] == check_box]
        if not found.empty:
            r = int(found.iloc[0]['rank'])
            pct = float(found.iloc[0]['rank_pct'])
            checked_msg = f'✅ Winner BOX {check_box} IS in Top-N. Global rank #{r} (percentile {pct:.1f}%). Best straight suggestion: {found.iloc[0]["best_straight"]}'
        else:
            where = ranked[ranked['box'] == check_box]
            if where.empty:
                checked_msg = f'❌ Winner BOX {check_box} was not generated by the grid (it does not appear in the full ranked list).'
            else:
                r = int(where.iloc[0]['rank'])
                pct = float(where.iloc[0]['rank_pct'])
                checked_msg = f'⚠️ Winner BOX {check_box} was generated but not in Top-N. Global rank #{r} (percentile {pct:.1f}%). Increase Top-N to at least {r} to include it.'

    with c1:
        if checked_msg:
            st.info(checked_msg)
        show_cols = ['rank', 'rank_pct', 'box', 'best_straight', 'score']
        st.dataframe(topn[show_cols], use_container_width=True, height=520)

    st.subheader('Straight-play recommendations')
    st.write('For each box combo, the app recommends the single most likely straight ordering (based on position frequencies + inclusion boosts).')
    st.dataframe(topn[['box', 'best_straight', 'rank', 'rank_pct']], use_container_width=True)

    with st.expander('Show FULL ranked list (all generated boxes)'):
        st.dataframe(ranked[['rank', 'rank_pct', 'box', 'best_straight', 'score']], use_container_width=True, height=520)


if __name__ == '__main__':
    main()
