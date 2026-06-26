"""
ACCDM-2 (Accelerating Dual Momentum v2)
========================================
Strategie momentum dual sur SPY/QQQ avec filtre force et levier x1.5.

Univers
-------
- SPY, QQQ : actifs detenus
- BIL : actif de comparaison (signal)
- Cash : remunere a 2%/an quand strategie hors marche

Logique
-------
1. Score mensuel = perf 1m + perf 3m + perf 6m sur SPY et BIL
2. Si score(SPY) >= score(BIL) : allocation 50/50 SPY+QQQ
   Sinon : 100% cash
3. Filtre force : chaque mercredi, si perf 15j du portefeuille 50/50 < 0%,
   sortie complete en cash jusqu'au prochain rebalancement mensuel
4. Levier x1.5 applique sur les rendements (cout de financement 3%/an)

Parametres figes (issus chantier IS 2000-2018, valides OOS 2018-2024)
---------------------------------------------------------------------
- LOOKBACKS_SCORE = [1, 3, 6] mois
- LOOKBACK_FILTRE = 15 jours
- SEUIL_FILTRE = 0.0
- LEVIER = 1.5
- COST_FINANCING = 0.03 / an

Statut : VALIDEE (Sharpe OOS / Sharpe IS = 1.12)
Date validation : 1 mai 2026
Date demarrage live : 4 mai 2026
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass


# ============================================================
# PARAMETRES FIGES (NE PAS MODIFIER)
# ============================================================

# Univers
ASSETS_HELD = ["SPY", "QQQ"]
ASSET_SIGNAL = "BIL"
ASSETS_ALL = ["SPY", "QQQ", "BIL"]

# Score
LOOKBACKS_SCORE = [1, 3, 6]

# Filtre force
LOOKBACK_FILTRE = 15
SEUIL_FILTRE = 0.0
JOUR_FILTRE = 2  # mercredi
JOUR_FILTRE_FALLBACK = 3  # jeudi

# Allocation
ALLOC_RISKY = {"SPY": 0.5, "QQQ": 0.5}

# Levier
LEVIER = 1.5
COST_FINANCING_ANNUAL = 0.03

# Cash
CASH_YIELD_ANNUAL = 0.02
CASH_YIELD_DAILY = (1 + CASH_YIELD_ANNUAL) ** (1 / 252) - 1

# Couts transaction
COST_BPS = 3 / 10_000

# Capital initial
INITIAL_CAPITAL = 10_000_000

# Dates cle
LIVE_START_DATE = pd.Timestamp("2026-05-04")


# ============================================================
# CHARGEMENT DATA
# ============================================================

def load_data(data_dir: str | Path, end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Charge les CSV SPY, QQQ, BIL et retourne un DataFrame aligne."""
    data_dir = Path(data_dir)
    series = {}
    for name in ASSETS_ALL:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if "Close" not in df.columns:
            raise ValueError(f"Colonne 'Close' manquante dans {path}")
        series[name] = df["Close"].rename(name)
    prices = pd.concat(series.values(), axis=1).sort_index().ffill().dropna()
    if end_date is not None:
        prices = prices.loc[prices.index <= end_date]
    return prices


# ============================================================
# CALCUL DU SCORE ET DE L'ALLOCATION CIBLE
# ============================================================

def compute_monthly_score(prices_eom: pd.DataFrame) -> pd.DataFrame:
    """Score momentum (1m + 3m + 6m) sur prix de fin de mois."""
    score = pd.DataFrame(index=prices_eom.index, columns=[ASSETS_HELD[0], ASSET_SIGNAL], dtype=float)
    for asset in ["SPY", "BIL"]:
        s = prices_eom[asset]
        total = pd.Series(0.0, index=s.index)
        for lb in LOOKBACKS_SCORE:
            total = total + s.pct_change(lb)
        score[asset] = total
    return score


def compute_target_allocation(score: pd.DataFrame) -> pd.DataFrame:
    """Allocation cible : SPY > BIL -> 50/50, sinon CASH 100%."""
    alloc = pd.DataFrame(0.0, index=score.index, columns=["SPY", "QQQ", "CASH"])
    for date, row in score.iterrows():
        if row.isna().any():
            alloc.loc[date, "CASH"] = 1.0
            continue
        if row["SPY"] >= row["BIL"]:
            alloc.loc[date, "SPY"] = ALLOC_RISKY["SPY"]
            alloc.loc[date, "QQQ"] = ALLOC_RISKY["QQQ"]
        else:
            alloc.loc[date, "CASH"] = 1.0
    return alloc


# ============================================================
# DETERMINATION DES JOURS DE TRIGGER FILTRE
# ============================================================

def get_filter_days(prices_index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Retourne l'ensemble des dates ou le filtre force doit etre evalue (mercredi sinon jeudi)."""
    df = pd.DataFrame({"d": prices_index})
    df["weekday"] = df["d"].dt.weekday
    df["week"] = df["d"].dt.isocalendar().week
    df["year"] = df["d"].dt.isocalendar().year
    days = set()
    for (_, _), grp in df.groupby(["year", "week"]):
        wed = grp[grp["weekday"] == JOUR_FILTRE]
        if len(wed) > 0:
            days.add(wed.iloc[0]["d"])
        else:
            thu = grp[grp["weekday"] == JOUR_FILTRE_FALLBACK]
            if len(thu) > 0:
                days.add(thu.iloc[0]["d"])
    return days


# ============================================================
# EVALUATION DU FILTRE FORCE
# ============================================================

def evaluate_filter(prices: pd.DataFrame, date: pd.Timestamp, positions: dict[str, float]) -> bool:
    """True si le filtre force se declenche (sortie cash requise)."""
    if positions.get("SPY", 0) <= 0 or positions.get("QQQ", 0) <= 0:
        return False
    idx_today = prices.index.get_loc(date)
    idx_lb = idx_today - LOOKBACK_FILTRE
    if idx_lb < 0:
        return False
    date_lb = prices.index[idx_lb]
    p_today = prices.loc[date]
    p_lb = prices.loc[date_lb]
    p_now = 0.5 * p_today["SPY"] + 0.5 * p_today["QQQ"]
    p_old = 0.5 * p_lb["SPY"] + 0.5 * p_lb["QQQ"]
    perf = p_now / p_old - 1
    return perf < SEUIL_FILTRE


# ============================================================
# BACKTEST V1 (rétro-compatibilité, conservé pour comparaison)
# ============================================================

@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    trades: pd.DataFrame
    equity_levered: pd.Series


def run_backtest(
    prices: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    apply_leverage: bool = True,
) -> BacktestResult:
    """Backtest standalone (levier appliqué a posteriori). NE PAS UTILISER pour levier > 1.
    Utiliser plutôt run_backtest_v2 (dans scripts/backtest_v2.py)."""
    prices_eom = prices.resample("ME").last()
    score = compute_monthly_score(prices_eom)
    alloc_target = compute_target_allocation(score)
    alloc_target = alloc_target.loc[score.dropna().index]

    rebal_dates = []
    valid_idx = []
    for d in alloc_target.index:
        cands = prices.index[prices.index <= d]
        if len(cands) > 0:
            rebal_dates.append(cands[-1])
            valid_idx.append(d)
    alloc_bt = alloc_target.loc[valid_idx].copy()
    alloc_bt.index = pd.DatetimeIndex(rebal_dates)
    alloc_bt = alloc_bt[~alloc_bt.index.duplicated(keep="last")]

    prices_bt = prices.loc[prices.index >= alloc_bt.index.min()].copy()

    cash = float(initial_capital)
    positions = {a: 0.0 for a in ASSETS_HELD}
    trades_list = []
    eq_list = []
    rebal_set = set(alloc_bt.index)
    prev_target = None
    filter_days = get_filter_days(prices_bt.index)
    forced_cash = False

    for date in prices_bt.index:
        p = prices_bt.loc[date]
        cash *= 1 + CASH_YIELD_DAILY

        if date in rebal_set:
            target = alloc_bt.loc[date].to_dict()
            target_changed = (prev_target is None) or any(
                abs(target[k] - prev_target.get(k, 0.0)) > 1e-9 for k in target
            )
            need_rebal = target_changed or (
                sum(positions.values()) < 1e-6
                and any(target.get(a, 0) > 0 for a in ASSETS_HELD)
            )
            if need_rebal:
                pv = sum(positions[a] * p[a] for a in ASSETS_HELD)
                eq_t = cash + pv
                tq = {a: int((eq_t * target.get(a, 0)) / p[a]) for a in ASSETS_HELD}
                for a in ASSETS_HELD:
                    dq = tq[a] - positions[a]
                    if dq < 0:
                        sq = abs(dq)
                        val = sq * p[a]
                        cost = val * COST_BPS
                        cash += val - cost
                        trades_list.append({
                            "date": date, "asset": a, "side": "SELL",
                            "qty": sq, "price": p[a], "reason": "REBAL",
                        })
                        positions[a] = tq[a]
                buys = [(a, tq[a] - positions[a]) for a in ASSETS_HELD if tq[a] - positions[a] > 0]
                if buys:
                    total = sum(q * p[a] for a, q in buys)
                    avail = cash / (1 + COST_BPS)
                    if total <= avail + 1e-6:
                        for a, qty in buys:
                            val = qty * p[a]
                            cost = val * COST_BPS
                            cash -= val + cost
                            trades_list.append({
                                "date": date, "asset": a, "side": "BUY",
                                "qty": qty, "price": p[a], "reason": "REBAL",
                            })
                            positions[a] = tq[a]
                    else:
                        scale = avail / total
                        for a, qty in buys:
                            qs = int(qty * scale)
                            if qs > 0:
                                val = qs * p[a]
                                cost = val * COST_BPS
                                cash -= val + cost
                                trades_list.append({
                                    "date": date, "asset": a, "side": "BUY",
                                    "qty": qs, "price": p[a], "reason": "REBAL",
                                })
                                positions[a] += qs
                prev_target = target.copy()
                forced_cash = False

        if date in filter_days and not forced_cash:
            if evaluate_filter(prices_bt, date, positions):
                for a in ASSETS_HELD:
                    if positions[a] > 0:
                        sq = positions[a]
                        val = sq * p[a]
                        cost = val * COST_BPS
                        cash += val - cost
                        trades_list.append({
                            "date": date, "asset": a, "side": "SELL",
                            "qty": sq, "price": p[a], "reason": "FORCE_EXIT",
                        })
                        positions[a] = 0
                forced_cash = True

        pv = sum(positions[a] * p[a] for a in ASSETS_HELD)
        eq_list.append({"date": date, "equity": cash + pv, "cash": cash})

    eq_df = pd.DataFrame(eq_list).set_index("date")
    trades_df = pd.DataFrame(trades_list)

    eq_lev = None
    if apply_leverage:
        eq_lev = apply_leverage_to_equity(eq_df["equity"])

    return BacktestResult(
        equity=eq_df["equity"],
        cash=eq_df["cash"],
        trades=trades_df,
        equity_levered=eq_lev if eq_lev is not None else eq_df["equity"].copy(),
    )


def apply_leverage_to_equity(equity: pd.Series, levier: float = LEVIER,
                              cost_financing_annual: float = COST_FINANCING_ANNUAL) -> pd.Series:
    """[DEPRECATED V1] Applique un levier multiplicateur sur les rendements quotidiens.
    
    ATTENTION : Ne respecte pas pleinement le protocole §1.2 pour levier > 1.
    Utiliser run_backtest_v2 dans scripts/backtest_v2.py pour la version conforme.
    
    Conservée pour rétro-compatibilité uniquement.
    """
    ret = equity.pct_change().fillna(0)
    cost_daily = (levier - 1) * cost_financing_annual / 252
    ret_lev = ret * levier - cost_daily
    eq_lev = (1 + ret_lev).cumprod() * equity.iloc[0]
    return eq_lev


# ============================================================
# METRIQUES
# ============================================================

def compute_metrics(equity: pd.Series) -> dict:
    """Metriques standard sur une serie d'equity."""
    if len(equity) < 2:
        return {}
    ret = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol > 0 else np.nan
    rmax = equity.cummax()
    dd = equity / rmax - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    ann = equity.resample("YE").last().pct_change().dropna()
    pct_pos = (ann > 0).sum() / len(ann) * 100 if len(ann) > 0 else np.nan
    return {
        "CAGR": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "AnneesPos": round(pct_pos, 1) if not np.isnan(pct_pos) else np.nan,
        "Volatility": round(vol * 100, 2),
        "n_years": round(n_years, 2),
        "capital_final_M": round(equity.iloc[-1] / 1e6, 2),
    }


# ============================================================
# FONCTIONS LIVE
# ============================================================

def get_current_signal(prices: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict:
    """Retourne le signal courant : allocation cible + statut filtre."""
    if as_of is None:
        as_of = prices.index[-1]
    else:
        as_of = pd.Timestamp(as_of)

    prices_to = prices.loc[prices.index <= as_of]
    prices_eom = prices_to.resample("ME").last()
    score = compute_monthly_score(prices_eom)
    last_score = score.dropna().iloc[-1]
    score_spy = last_score["SPY"]
    score_bil = last_score["BIL"]
    monthly_signal = "RISKY" if score_spy >= score_bil else "CASH"

    perf_15d = None
    filter_triggered = False
    idx = prices_to.index.get_loc(as_of)
    if idx >= LOOKBACK_FILTRE:
        date_lb = prices_to.index[idx - LOOKBACK_FILTRE]
        p_now = 0.5 * prices_to.loc[as_of, "SPY"] + 0.5 * prices_to.loc[as_of, "QQQ"]
        p_old = 0.5 * prices_to.loc[date_lb, "SPY"] + 0.5 * prices_to.loc[date_lb, "QQQ"]
        perf_15d = p_now / p_old - 1
        filter_triggered = perf_15d < SEUIL_FILTRE

    if monthly_signal == "CASH":
        target = {"SPY": 0.0, "QQQ": 0.0, "CASH": 1.0}
    elif filter_triggered:
        target = {"SPY": 0.0, "QQQ": 0.0, "CASH": 1.0}
    else:
        target = {"SPY": 0.5, "QQQ": 0.5, "CASH": 0.0}

    return {
        "as_of": as_of,
        "monthly_signal": monthly_signal,
        "score_spy": float(score_spy),
        "score_bil": float(score_bil),
        "score_diff": float(score_spy - score_bil),
        "filter_triggered": filter_triggered,
        "perf_15d": float(perf_15d) if perf_15d is not None else None,
        "target_allocation": target,
    }


def is_monthly_signal_day(date: pd.Timestamp, prices_index: pd.DatetimeIndex) -> bool:
    date = pd.Timestamp(date)
    next_month = date + pd.Timedelta(days=1)
    while next_month.month == date.month:
        if next_month in prices_index:
            return False
        next_month += pd.Timedelta(days=1)
    return date in prices_index


def is_filter_day(date: pd.Timestamp, prices_index: pd.DatetimeIndex) -> bool:
    date = pd.Timestamp(date)
    return date in get_filter_days(prices_index)


# ============================================================
# HISTORIQUE DES ALLOCATIONS
# ============================================================

def compute_allocation_history(prices: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit l'historique des allocations effectives (en %)."""
    result = run_backtest(prices, apply_leverage=False)
    eq = result.equity
    cash = result.cash

    trades = result.trades.copy()
    if len(trades) == 0:
        return pd.DataFrame()
    trades["date"] = pd.to_datetime(trades["date"])

    positions = {a: pd.Series(0.0, index=eq.index) for a in ASSETS_HELD}
    current_pos = {a: 0.0 for a in ASSETS_HELD}

    for date in eq.index:
        day_trades = trades[trades["date"] == date]
        for _, t in day_trades.iterrows():
            asset = t["asset"]
            if t["side"] == "BUY":
                current_pos[asset] += t["qty"]
            else:
                current_pos[asset] -= t["qty"]
                if current_pos[asset] < 0:
                    current_pos[asset] = 0
        for a in ASSETS_HELD:
            positions[a].loc[date] = current_pos[a]

    history = pd.DataFrame(index=eq.index)
    for a in ASSETS_HELD:
        history[f"value_{a}"] = positions[a] * prices[a].reindex(eq.index)
    history["value_cash"] = cash
    history["total"] = history[[f"value_{a}" for a in ASSETS_HELD]].sum(axis=1) + history["value_cash"]

    history["poids_spy"] = history["value_SPY"] / history["total"] * 100
    history["poids_qqq"] = history["value_QQQ"] / history["total"] * 100
    history["poids_cash"] = history["value_cash"] / history["total"] * 100

    history["phase"] = (
        (history["value_SPY"] > 0) | (history["value_QQQ"] > 0)
    ).map({True: "MARCHE", False: "CASH"})

    return history[["poids_spy", "poids_qqq", "poids_cash", "phase"]]
