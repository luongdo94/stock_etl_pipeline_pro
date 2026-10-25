"""Pure Market Pulse context: estimated ETF creations/redemptions and factual rotation warnings."""
import pandas as pd

from etl.sector_rotation import SECTOR_ETFS, BENCHMARK

DEFENSIVE = ("Utilities", "Consumer Staples", "Healthcare")
CYCLICAL = ("Technology", "Consumer Discretionary", "Industrials", "Financials", "Communication Services")


def estimate_flows(snapshots):
    """Calendar 1W/1M, using the last snapshot on/before the start; missing or unchanged is unknown."""
    rows = []
    if snapshots.empty:
        return pd.DataFrame()
    for ticker, history in snapshots.groupby("ticker"):
        h = history.sort_values("as_of").reset_index(drop=True).copy()
        h["as_of"] = pd.to_datetime(h["as_of"])
        last = h.iloc[-1]
        start = h.iloc[0]["as_of"].date().isoformat()
        row = dict(ticker=ticker, sector=SECTOR_ETFS.get(ticker, BENCHMARK),
                   as_of=last.as_of, history_start=start)
        derived = "shares_source" in h and (h["shares_source"] == "aum/nav").any()
        changed = h["shares_outstanding"].ne(last.shares_outstanding)
        streak_start = int(h.index[changed].max()) + 1 if changed.any() else 0
        since = h.iloc[min(streak_start, len(h) - 1)].as_of.date().isoformat()
        for period, offset, slack in (("1w", pd.Timedelta(days=7), pd.Timedelta(days=3)),
                                      ("1m", pd.DateOffset(months=1), pd.Timedelta(days=7))):
            target = last.as_of - offset
            old = h[h.as_of <= target]
            usd = pct = float("nan")
            note = f"flow history starts {start}"
            if not old.empty and old.iloc[-1].as_of < target - slack:
                # K5-08: the nearest snapshot is too old to be a 1W / 1M base — the change spans a longer period
                note = f"no snapshot near {target:%Y-%m-%d} (nearest {old.iloc[-1].as_of:%Y-%m-%d})"
            elif not old.empty:
                base = old.iloc[-1]
                note = ""
                needed = [last.shares_outstanding, base.shares_outstanding, last.price_close, base.total_assets]
                if all(pd.notna(v) and 0 < v < float("inf") for v in needed):
                    delta = last.shares_outstanding - base.shares_outstanding
                    ratio = last.shares_outstanding / base.shares_outstanding
                    if delta == 0:
                        note = f"provider figure unchanged since {since}"
                    elif ratio >= 1.5 or ratio <= 1 / 1.5:
                        # K5-09: a 50 %+ jump in shares within the period looks like a split, not creations
                        note = f"share count changed x{ratio:.2f} — a split cannot be ruled out"
                    else:
                        usd = delta * last.price_close
                        pct = usd / base.total_assets * 100
                else:
                    note = "snapshot figures unavailable"
            if derived:
                note = "; ".join(filter(None, [note, "shares derived from total assets / NAV"]))
            row.update({f"flow_{period}_usd": usd, f"flow_{period}_pct": pct, f"flow_{period}_note": note})
        rows.append(row)
    return pd.DataFrame(rows)


def rotation_warnings(rotation, rules, flows=None):
    """No I/O, scoring or recommendations. Unknown inputs cannot satisfy a rule."""
    out = []
    if rotation.empty or "sector" not in rotation:
        return out
    s = rotation.drop_duplicates("sector").set_index("sector").reindex(list(SECTOR_ETFS.values()))
    s = s.reindex(columns=["perf_1w", "perf_1m"]).apply(pd.to_numeric, errors="coerce")
    s = s.where(s.abs() < float("inf"))

    def add(kind, sectors, message, severity="info"):
        out.append(dict(kind=kind, sectors=list(sectors), message=f"{kind}: {message}", severity=severity))

    falling = s.index[s.perf_1w < 0].tolist()
    if len(falling) >= rules["broad_weakness_count"]:
        add("BROAD WEAKNESS", falling, f"{len(falling)}/11 sectors declined over 1W.", "warning")
    bench = rotation.loc[rotation.sector == BENCHMARK].reindex(columns=["perf_1w", "perf_1m"])
    if not bench.empty:
        b = pd.to_numeric(bench.iloc[0], errors="coerce")
        rel = s - b.where(b.abs() < float("inf"))
        groups = rel.loc[[*DEFENSIVE, *CYCLICAL], "perf_1m"]
        if groups.notna().all():
            gap = groups.loc[list(DEFENSIVE)].mean() - groups.loc[list(CYCLICAL)].mean()
            if abs(gap) > rules["group_gap_pp"]:
                defensive = gap > 0
                add("DEFENSIVE ROTATION" if defensive else "CYCLICAL ROTATION", groups.index,
                    f"{'Defensive' if defensive else 'Cyclical'} sectors led by {abs(gap):.1f} pp vs SPY over 1M "
                    f"({'risk-off' if defensive else 'risk-on'} pattern).", "warning" if defensive else "info")
        if rel.perf_1m.notna().all():
            ranked = rel.perf_1m.sort_values(ascending=False, kind="stable")
            # A flat/tied ranking does not establish leadership or lagging.
            for names, kind, sign in ((ranked.index[:3], "LEADER FADING", -1),
                                      (ranked.index[-3:], "LAGGARD TURNING", 1)):
                for sector in names:
                    month, week = rel.loc[sector, ["perf_1m", "perf_1w"]]
                    boundary = ranked.iloc[3] if sign == -1 else ranked.iloc[-4]
                    if sign * week > rules["reversal_pp"] and (month > boundary if sign == -1 else month < boundary):
                        add(kind, [sector], f"{sector}: 1M {month:+.1f} pp vs SPY; 1W {week:+.1f} pp vs SPY.",
                            "warning" if sign == -1 else "info")
    if flows is not None and not flows.empty and "sector" in flows:
        for _, f in flows[flows.sector.isin(s.index)].iterrows():
            week, month = f.get("flow_1w_pct", float("nan")), f.get("flow_1m_pct", float("nan"))
            if pd.notna(week) and week < -rules["flow_aum_pct"]:
                add("FLOW", [f.sector], f"{f.sector}: estimated 1W outflow {week:.1f}% of starting AUM.", "warning")
            if pd.notna(month) and month > rules["flow_aum_pct"] and s.loc[f.sector, "perf_1m"] < 0:
                add("FLOW", [f.sector], f"{f.sector}: estimated 1M inflow {month:.1f}% of starting AUM while 1M price declined.")
    return out
