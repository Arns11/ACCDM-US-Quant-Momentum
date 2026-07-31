"""
alert_email.py — Alerte email US Index Momentum (Quant Signals)
================================================================
VERSION 2.0 — 31 juillet 2026

Corrections apportees par rapport a la version precedente :

  1. GARDE JOUR DE SIGNAL
     Le programme verifie lui-meme qu'on est bien un jour de signal
     (mercredi de filtre, ou dernier jour ouvre du mois). Sinon il ne
     fait rien. Fini les 4 emails d'affilee en fin de mois.

  2. CONTROLE DE FRAICHEUR
     Le programme verifie que la derniere donnee recue correspond bien
     a la seance du jour. Si les donnees ont du retard, il refuse
     d'envoyer et previent par email technique.

  3. SOURCE DE SECOURS
     EODHD en premier. Si indisponible, bascule automatique sur Yahoo
     Finance. Si les deux echouent, email technique d'alerte.

  4. CLE API
     Plus aucune cle ecrite dans le code. Lecture par variable
     d'environnement uniquement.

  5. JOURNAL DES SIGNAUX
     Chaque email envoye laisse une ligne dans signals_log.csv.
     Permet de verifier a posteriori ce qui a reellement ete diffuse.

  6. REENVOI SMTP
     3 tentatives espacees de 30, 60 et 120 secondes.

Variables d'environnement attendues :
    GMAIL_USER, GMAIL_APP_PASS, ALERT_RECIPIENTS, EODHD_API_KEY
Facultatif :
    ALERT_RECIPIENTS_TECH   (destinataire des alertes techniques,
                             defaut = GMAIL_USER)
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import smtplib
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# --- Import de la strategie (racine, src/, ou dossier courant) -------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "src"), os.path.join(_HERE, ".."),
           os.path.join(_HERE, "..", "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.strategy import (
        ASSETS_HELD, COST_BPS, CASH_YIELD_DAILY, LIVE_START_DATE,
        COST_FINANCING_ANNUAL,
        compute_monthly_score, compute_target_allocation,
        get_filter_days, evaluate_filter, get_current_signal,
    )
except ImportError:
    from strategy import (
        ASSETS_HELD, COST_BPS, CASH_YIELD_DAILY, LIVE_START_DATE,
        COST_FINANCING_ANNUAL,
        compute_monthly_score, compute_target_allocation,
        get_filter_days, evaluate_filter, get_current_signal,
    )


# ==========================================================================
# CONFIGURATION
# ==========================================================================

SMTP_SERVER, SMTP_PORT = "smtp.gmail.com", 587
CAPITAL_EXEMPLE = 10_000
BRAND = "Quant Signals"
STRAT_NAME = "US Index Momentum"

ROOT = Path(_HERE).resolve()
if ROOT.name in ("scripts", "src"):
    ROOT = ROOT.parent
SIGNALS_LOG = ROOT / "signals_log.csv"
ETAT_JSON = ROOT / "etat_strategie.json"

# Capital de reference affiche sur la page publique. Tout est en dollars :
# la strategie achete des actifs cotes en dollars, on n'introduit aucune
# conversion de devise qui melangerait performance et taux de change.
CAPITAL_REFERENCE_USD = 10_000
DEVISE = "USD"

# Taux de financement annuel applique a la courbe levier 1,5 de l'email.
# Decision Arnaud du 31/07/2026 : 2,5 %/an (au lieu des 3 % du backtest).
FINANCEMENT_LEVIER_EMAIL = 0.025

SMTP_RETRY_DELAYS = [30, 60, 120]
NY = ZoneInfo("America/New_York")
US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# Heure de cloture des marches US, en heure de New York.
# On ajoute une marge de securite avant de considerer la seance close.
US_CLOSE_HOUR = 16
US_CLOSE_MARGIN_MIN = 20


def get_api_key() -> str:
    """Cle EODHD lue uniquement dans l'environnement. Aucune valeur en dur."""
    key = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_KEY")
    if not key:
        raise RuntimeError(
            "Cle EODHD absente. Definir la variable d'environnement "
            "EODHD_API_KEY (secret GitHub ou fichier .env local)."
        )
    return key


# ==========================================================================
# TELECHARGEMENT DES DONNEES — EODHD puis Yahoo en secours
# ==========================================================================

def _eodhd_series(symbol: str, api_key: str, frm: str = "1999-01-01") -> pd.DataFrame:
    url = f"https://eodhd.com/api/eod/{symbol}"
    r = requests.get(
        url,
        params={"api_token": api_key, "fmt": "json", "from": frm, "period": "d"},
        timeout=60,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError(f"EODHD : aucune donnee pour {symbol}")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def _assemble(spy_close, spy_open, qqq_close, qqq_open, bil_close, irx_close):
    """Assemble les series en deux tableaux : cloture et ouverture ajustees.

    BIL n'existe qu'a partir de 2007. Avant cette date, on reconstitue une
    serie equivalente a partir du taux monetaire IRX.
    """
    close = pd.DataFrame({"SPY": spy_close, "QQQ": qqq_close})
    open_adj = pd.DataFrame({"SPY": spy_open, "QQQ": qqq_open})

    bil_real = bil_close.dropna().copy()
    splice = bil_real.index.min()
    irx_pre = irx_close.loc[irx_close.index < splice].dropna()
    if len(irx_pre) > 0:
        daily_ret = (1 + irx_pre / 100.0) ** (1 / 252) - 1
        synth = (1 + daily_ret).cumprod()
        synth = synth * (bil_real.iloc[0] / synth.iloc[-1])
        bil_full = pd.concat([synth, bil_real]).sort_index()
    else:
        bil_full = bil_real
    bil_full = bil_full[~bil_full.index.duplicated(keep="last")]
    close["BIL"] = bil_full.reindex(close.index).ffill()

    close = close.sort_index().ffill().dropna()
    open_adj = open_adj.reindex(close.index)
    return close, open_adj


def fetch_from_eodhd():
    api_key = get_api_key()
    spy = _eodhd_series("SPY.US", api_key)
    qqq = _eodhd_series("QQQ.US", api_key)
    bil = _eodhd_series("BIL.US", api_key)
    irx = _eodhd_series("IRX.INDX", api_key)

    return _assemble(
        spy_close=spy["adjusted_close"],
        spy_open=spy["open"] * (spy["adjusted_close"] / spy["close"]),
        qqq_close=qqq["adjusted_close"],
        qqq_open=qqq["open"] * (qqq["adjusted_close"] / qqq["close"]),
        bil_close=bil["adjusted_close"],
        irx_close=irx["close"],
    )


def fetch_from_yahoo():
    import yfinance as yf

    def dl(ticker):
        df = yf.download(ticker, start="1999-01-01", progress=False,
                         auto_adjust=True, threads=False)
        if df is None or df.empty:
            raise RuntimeError(f"Yahoo : aucune donnee pour {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    spy, qqq, bil, irx = dl("SPY"), dl("QQQ"), dl("BIL"), dl("^IRX")

    return _assemble(
        spy_close=spy["Close"], spy_open=spy["Open"],
        qqq_close=qqq["Close"], qqq_open=qqq["Open"],
        bil_close=bil["Close"], irx_close=irx["Close"],
    )


def build_frames():
    """Renvoie (cloture, ouverture, nom de la source utilisee)."""
    try:
        close, open_adj = fetch_from_eodhd()
        return close, open_adj, "EODHD"
    except Exception as e:
        print(f"  EODHD indisponible : {type(e).__name__} : {e}")
        print("  Bascule sur Yahoo Finance ...")
        close, open_adj = fetch_from_yahoo()
        return close, open_adj, "Yahoo Finance (secours)"


# ==========================================================================
# CONTROLE DE FRAICHEUR
# ==========================================================================

def last_expected_session(now_ny: datetime | None = None) -> pd.Timestamp:
    """Derniere seance US qui devrait etre disponible a cet instant.

    Si la cloture du jour n'a pas encore eu lieu (ou vient tout juste
    d'avoir lieu), on attend la seance ouvree precedente.
    """
    if now_ny is None:
        now_ny = datetime.now(NY)

    today = pd.Timestamp(now_ny.date())
    close_time = now_ny.replace(hour=US_CLOSE_HOUR, minute=US_CLOSE_MARGIN_MIN,
                                second=0, microsecond=0)

    is_bday = len(pd.bdate_range(today, today, freq=US_BDAY)) == 1
    if is_bday and now_ny >= close_time:
        return today
    return (today - US_BDAY).normalize()


def is_last_bday_of_month(ts: pd.Timestamp) -> bool:
    """Vrai si cette date est le dernier jour ouvre de son mois.

    S'appuie sur le calendrier des jours feries americains, et non sur
    la presence ou non de donnees apres cette date. C'est indispensable :
    les donnees s'arretent forcement au jour courant, ce qui ferait
    passer n'importe quel jour pour une fin de mois.
    """
    ts = pd.Timestamp(ts).normalize()
    first = pd.Timestamp(ts.year, ts.month, 1)
    last = first + pd.offsets.MonthEnd(1)
    bdays = pd.bdate_range(start=first, end=last, freq=US_BDAY)
    return len(bdays) > 0 and ts == bdays[-1].normalize()


def check_freshness(close: pd.DataFrame) -> tuple[bool, str]:
    """Verifie que la derniere donnee correspond bien a la seance attendue."""
    last_data = pd.Timestamp(close.index[-1]).normalize()
    expected = last_expected_session()
    if last_data >= expected:
        return True, f"Donnees a jour ({last_data.date()})"
    retard = (expected - last_data).days
    return False, (
        f"Donnees en retard : derniere seance recue {last_data.date()}, "
        f"seance attendue {expected.date()} (retard de {retard} jour(s))."
    )


# ==========================================================================
# GARDE : EST-CE UN JOUR DE SIGNAL ?
# ==========================================================================

def is_signal_day(close: pd.DataFrame) -> tuple[bool, str]:
    """Determine si la derniere seance est un jour de signal.

    Deux cas donnent lieu a un envoi :
      - jour de filtre hebdomadaire (mercredi, ou jeudi si mercredi absent)
      - dernier jour ouvre du mois (rebalancement mensuel)
    """
    last = pd.Timestamp(close.index[-1]).normalize()
    idx = close.index

    filter_days = get_filter_days(idx)
    is_filter = last in filter_days

    is_month_end = is_last_bday_of_month(last)

    if is_filter and is_month_end:
        return True, "filtre hebdomadaire + rebalancement mensuel"
    if is_filter:
        return True, "filtre hebdomadaire"
    if is_month_end:
        return True, "rebalancement mensuel"
    return False, "jour ordinaire, aucun signal prevu"


# ==========================================================================
# MOTEUR — rejoue la strategie en mode C (ordre a l'ouverture du lendemain)
# ==========================================================================

def run_modeC(prices, prices_open, leverage=1.0, initial_capital=10_000_000):
    prices_eom = prices.resample("ME").last()
    score = compute_monthly_score(prices_eom)
    alloc_target = compute_target_allocation(score).loc[score.dropna().index]

    last_data = pd.Timestamp(prices.index[-1])

    rebal_dates, valid_idx = [], []
    for d in alloc_target.index:
        cands = prices.index[prices.index <= d]
        if not len(cands):
            continue
        mapped = cands[-1]
        if mapped.month != d.month:
            continue

        # Mois en cours : on n'accepte le rebalancement QUE si la seance
        # est reellement le dernier jour ouvre du mois. Sans cette regle,
        # les donnees s'arretant au jour courant, chaque jour du mois
        # passerait pour une fin de mois et regenererait un ordre d'achat.
        is_current_month = (d.year == last_data.year and d.month == last_data.month)
        if is_current_month:
            if not is_last_bday_of_month(mapped):
                continue
        else:
            if (d - mapped).days > 4:
                continue

        rebal_dates.append(mapped)
        valid_idx.append(d)
    alloc_bt = alloc_target.loc[valid_idx].copy()
    alloc_bt.index = pd.DatetimeIndex(rebal_dates)
    alloc_bt = alloc_bt[~alloc_bt.index.duplicated(keep="last")]
    prices_bt = prices.loc[prices.index >= alloc_bt.index.min()].copy()

    cash = float(initial_capital)
    positions = {a: 0.0 for a in ASSETS_HELD}
    trades, pending = [], []
    eq_list = []
    rebal_set = set(alloc_bt.index)
    prev_target = None
    filter_days = get_filter_days(prices_bt.index)
    forced_cash = False
    margin_breaches = 0

    for date in prices_bt.index:
        p = prices_bt.loc[date]

        if pending:
            try:
                po = prices_open.loc[date]
                for o in pending:
                    a, q = o["asset"], o["qty"]
                    px = float(po[a])
                    val = q * px
                    c = val * COST_BPS
                    if o["side"] == "BUY":
                        cash -= val + c
                        positions[a] += q
                    else:
                        cash += val - c
                        positions[a] -= q
                    trades.append({"date": date, "asset": a, "side": o["side"],
                                   "qty": q, "price": px, "reason": o["reason"]})
                pending = []
            except KeyError:
                pass

        if cash > 0:
            cash *= 1 + CASH_YIELD_DAILY
        elif cash < 0 and leverage > 1.0:
            # Interets quotidiens sur le cash emprunte (levier)
            cash -= (-cash) * FINANCEMENT_LEVIER_EMAIL / 252
            # Surveillance marge Reg-T (25% de la valeur des positions)
            pv_chk = sum(positions[a] * p[a] for a in ASSETS_HELD)
            if pv_chk > 0 and (cash + pv_chk) < 0.25 * pv_chk:
                margin_breaches += 1

        if date in rebal_set:
            target = alloc_bt.loc[date].to_dict()
            changed = (prev_target is None) or any(
                abs(target[k] - prev_target.get(k, 0)) > 1e-9 for k in target)
            empty = (sum(positions.values()) < 1e-6
                     and any(target.get(a, 0) > 0 for a in ASSETS_HELD))
            if changed or empty:
                eq = cash + sum(positions[a] * p[a] for a in ASSETS_HELD)
                tq = {a: int(eq * target.get(a, 0) * leverage / p[a])
                      if target.get(a, 0) > 0 else 0 for a in ASSETS_HELD}
                for a in ASSETS_HELD:
                    dq = tq[a] - positions[a]
                    if dq < 0:
                        pending.append({"asset": a, "side": "SELL",
                                        "qty": abs(dq), "reason": "ARBITRAGE"})
                    elif dq > 0:
                        pending.append({"asset": a, "side": "BUY",
                                        "qty": dq, "reason": "ARBITRAGE"})
                prev_target = target.copy()
                forced_cash = False

        if date in filter_days and not forced_cash:
            if evaluate_filter(prices_bt, date, positions):
                for a in ASSETS_HELD:
                    if positions[a] > 0:
                        pending.append({"asset": a, "side": "SELL",
                                        "qty": positions[a], "reason": "FORCE_EXIT"})
                forced_cash = True

        pv = sum(positions[a] * p[a] for a in ASSETS_HELD)
        eq_list.append({"date": date, "equity": cash + pv})

    equity = pd.DataFrame(eq_list).set_index("date")["equity"]
    last_trade = trades[-1] if trades else None
    return {"positions": positions, "pending": pending, "trades": trades,
            "last_trade": last_trade, "last_date": prices_bt.index[-1],
            "equity": equity, "margin_breaches": margin_breaches}


# ==========================================================================
# ETAT -> CONTEXTE DE L'EMAIL
# ==========================================================================

def compute_state(prices, prices_open):
    eng = run_modeC(prices, prices_open, leverage=1.0)
    sig = get_current_signal(prices)
    held_risky = sum(eng["positions"].values()) > 1e-6
    pending = eng["pending"]
    buys = any(o["side"] == "BUY" for o in pending)
    sells = any(o["side"] == "SELL" for o in pending)
    exit_reason = pending[0]["reason"] if (sells and pending) else None

    perf15 = sig["perf_15d"] * 100
    monthly = sig["monthly_signal"]
    filt = sig["filter_triggered"]
    spy_c, qqq_c = prices["SPY"].iloc[-1], prices["QQQ"].iloc[-1]
    date_str = eng["last_date"].strftime("%d/%m/%Y")

    last_trade = eng["last_trade"]
    last_action = None
    if last_trade:
        verb = "entree" if last_trade["side"] == "BUY" else "sortie"
        last_action = f"{verb} le {pd.Timestamp(last_trade['date']).strftime('%d/%m/%Y')}"

    if buys:
        action = "OUVRIR"
        directive = "Investir : 50% SPY et 50% QQQ"
        spy_pct, qqq_pct, cash_pct = 50, 50, 0
        why = (f"Le momentum mensuel est favorable (S&P 500 au-dessus du monetaire) et le "
               f"filtre de protection 15 jours est positif ({perf15:+.1f}%). La strategie se "
               f"repositionne sur le marche.")
    elif sells:
        action = "FERMER"
        directive = "Sortir du marche : passer 100% en liquidites"
        spy_pct, qqq_pct, cash_pct = 0, 0, 100
        if exit_reason == "FORCE_EXIT":
            why = (f"Le filtre de protection 15 jours s'est declenche : le portefeuille SPY/QQQ "
                   f"a recule de {perf15:+.1f}% sur 15 jours. La strategie sort entierement du "
                   f"marche par securite jusqu'au prochain signal mensuel.")
        else:
            why = ("Le momentum mensuel est repasse sous le monetaire. La strategie sort du "
                   "marche et se met en liquidites.")
    else:
        action = "MAINTENIR"
        if held_risky:
            directive = "Rien a faire : vous restez investi"
            spy_pct, qqq_pct, cash_pct = 50, 50, 0
            why = ("Aucun changement de signal cette semaine. Le momentum reste favorable et le "
                   "filtre de protection 15 jours n'est pas declenche. Conservez vos positions.")
        else:
            directive = "Rester a l'ecart : 100% liquidites"
            spy_pct, qqq_pct, cash_pct = 0, 0, 100
            if filt:
                why = (f"Le momentum mensuel reste favorable, mais le filtre de protection 15 jours "
                       f"est toujours declenche ({perf15:+.1f}%). La strategie reste hors marche "
                       f"jusqu'a ce que le filtre se debloque ou au prochain signal mensuel.")
            else:
                why = ("Le momentum mensuel est defavorable (S&P 500 sous le monetaire). La "
                       "strategie reste en liquidites en attendant un signal favorable.")

    filter_date = None
    if filt:
        if any(o["reason"] == "FORCE_EXIT" for o in pending):
            filter_date = eng["last_date"]
        else:
            fe = [t for t in eng["trades"] if t.get("reason") == "FORCE_EXIT"]
            if fe:
                filter_date = pd.Timestamp(fe[-1]["date"])
    if filt:
        fd = filter_date.strftime("%d/%m/%Y") if filter_date is not None else None
        filter_txt = f"DECLENCHE le {fd} ({perf15:+.1f}%)" if fd else f"DECLENCHE ({perf15:+.1f}%)"
    else:
        filter_txt = f"OK ({perf15:+.1f}%)"

    return dict(
        action=action, directive=directive, date=date_str,
        spy_pct=spy_pct, qqq_pct=qqq_pct, cash_pct=cash_pct, leverage=1,
        capital_exemple=CAPITAL_EXEMPLE, why=why, last_action=last_action,
        signal_date=eng["last_date"],
        spy_close=float(spy_c), qqq_close=float(qqq_c),
        _eng=eng,
        ctx_market=dict(
            mom_ok=(monthly == "RISKY"),
            mom_txt=("Favorable" if monthly == "RISKY" else "Defavorable")
                    + (" (neutralise par le filtre)" if (monthly == "RISKY" and filt) else ""),
            filter_on=filt,
            filter_txt=filter_txt,
            spy=f"{spy_c:,.2f}".replace(",", " "),
            qqq=f"{qqq_c:,.2f}".replace(",", " "),
        ),
    )


# ==========================================================================
# ETAT PUBLIC — fichier destine a la page du site
# ==========================================================================

def _derniere_operation(trades, side):
    """Renvoie les lignes du dernier mouvement d'achat ou de vente."""
    concernes = [t for t in trades if t["side"] == side]
    if not concernes:
        return []
    derniere_date = concernes[-1]["date"]
    return [t for t in concernes if t["date"] == derniere_date]


def compute_position_details(eng, prices, capital=CAPITAL_REFERENCE_USD):
    """Detaille la position courante ramenee au capital de reference.

    Sans effet de levier. Les quantites sont entieres, comme dans la
    realite : le reliquat non investi reste en liquidites.
    """
    investi = sum(eng["positions"].values()) > 1e-6
    cours = {a: float(prices[a].iloc[-1]) for a in ASSETS_HELD}

    if not investi:
        sorties = _derniere_operation(eng["trades"], "SELL")
        depuis = str(pd.Timestamp(sorties[0]["date"]).date()) if sorties else None
        return {
            "investi": False,
            "depuis": depuis,
            "lignes": [],
            "capital_reference": capital,
            "valeur_actuelle": round(capital, 2),
            "liquidites": round(capital, 2),
            "gain_latent": 0.0,
            "gain_latent_pct": 0.0,
        }

    achats = _derniere_operation(eng["trades"], "BUY")
    prix_entree = {t["asset"]: float(t["price"]) for t in achats}
    date_entree = str(pd.Timestamp(achats[0]["date"]).date()) if achats else None

    lignes, investi_total = [], 0.0
    for a in ASSETS_HELD:
        pe = prix_entree.get(a)
        if pe is None or pe <= 0:
            continue
        montant_cible = capital * 0.5
        qty = int(montant_cible / pe)
        cout = qty * pe
        valeur = qty * cours[a]
        investi_total += cout
        lignes.append({
            "actif": a,
            "quantite": qty,
            "prix_entree": round(pe, 4),
            "cours_actuel": round(cours[a], 4),
            "ecart_pct": round((cours[a] / pe - 1) * 100, 2),
            "montant_investi": round(cout, 2),
            "valeur_actuelle": round(valeur, 2),
            "gain_latent": round(valeur - cout, 2),
        })

    liquidites = capital - investi_total
    valeur_totale = sum(l["valeur_actuelle"] for l in lignes) + liquidites
    gain = valeur_totale - capital

    return {
        "investi": True,
        "depuis": date_entree,
        "lignes": lignes,
        "capital_reference": capital,
        "valeur_actuelle": round(valeur_totale, 2),
        "liquidites": round(liquidites, 2),
        "gain_latent": round(gain, 2),
        "gain_latent_pct": round(gain / capital * 100, 2),
    }


def prochaines_dates_signal(depuis: pd.Timestamp, horizon_jours: int = 45) -> dict:
    """Prochain rendez-vous hebdomadaire et prochaine fin de mois."""
    depuis = pd.Timestamp(depuis).normalize()
    prochain_filtre = None
    prochaine_fin_mois = None

    jour = depuis + timedelta(days=1)
    fin = depuis + timedelta(days=horizon_jours)
    while jour <= fin:
        ouvre = len(pd.bdate_range(jour, jour, freq=US_BDAY)) == 1
        if ouvre:
            if prochain_filtre is None and jour.weekday() == 2:
                prochain_filtre = jour
            if prochaine_fin_mois is None and is_last_bday_of_month(jour):
                prochaine_fin_mois = jour
        jour += timedelta(days=1)

    candidats = [d for d in (prochain_filtre, prochaine_fin_mois) if d is not None]
    return {
        "prochain_point_hebdomadaire": str(prochain_filtre.date()) if prochain_filtre is not None else None,
        "prochain_rebalancement_mensuel": str(prochaine_fin_mois.date()) if prochaine_fin_mois is not None else None,
        "prochaine_date_de_signal": str(min(candidats).date()) if candidats else None,
    }


def ecrire_etat_public(ctx, position, jour_de_signal, motif, source):
    """Ecrit le fichier lu par la page du site.

    Deux informations strictement separees :
      - l'etat de la position, valable en permanence
      - l'action a realiser, presente uniquement les jours de signal
    """
    date_signal = pd.Timestamp(ctx["signal_date"])

    if jour_de_signal:
        action = {
            "action_requise": ctx["action"] != "MAINTENIR",
            "type": ctx["action"],
            "consigne": ctx["directive"],
            "allocation_cible": {
                "SPY_pct": ctx["spy_pct"],
                "QQQ_pct": ctx["qqq_pct"],
                "liquidites_pct": ctx["cash_pct"],
            },
            "motif_du_jour": motif,
        }
    else:
        action = {
            "action_requise": False,
            "type": "AUCUNE",
            "consigne": "Aucune action a realiser aujourd'hui. "
                        "Ce n'est pas un jour de signal.",
            "allocation_cible": None,
            "motif_du_jour": "jour ordinaire",
        }

    etat = {
        "strategie": STRAT_NAME,
        "marque": BRAND,
        "devise": DEVISE,
        "capital_de_reference": CAPITAL_REFERENCE_USD,
        "levier": 1,
        "calcule_le": datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S"),
        "seance_de_reference": str(date_signal.date()),
        "source_donnees": source,
        "jour_de_signal": bool(jour_de_signal),
        "action": action,
        "position": position,
        "calendrier": prochaines_dates_signal(date_signal),
        "avertissement": "Information fournie a titre d'aide a la decision. "
                         "Ne constitue pas un conseil en investissement personnalise.",
    }

    ETAT_JSON.write_text(json.dumps(etat, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Etat public ecrit : {ETAT_JSON}")
    return etat


# ==========================================================================
# COURBE DE PERFORMANCE — image integree a l'email
# ==========================================================================

def max_drawdown_pct(equity: pd.Series) -> float:
    """Pire baisse depuis un sommet, en pourcentage negatif."""
    eq = equity.dropna()
    dd = eq / eq.cummax() - 1
    return float(dd.min() * 100)


def build_equity_png(equity: pd.Series, equity_lev: pd.Series | None = None) -> bytes:
    """Trace la courbe de la strategie, normalisee au capital de reference.

    Courbe principale : levier 1 (coherente avec le tableau d'allocation).
    Courbe d'arriere-plan facultative : levier 1,5, en gris leger, frais de
    financement inclus. Trait vertical au demarrage live, partie anterieure
    explicitement marquee comme simulee. Tout est en dollars.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    eq = equity.dropna()
    eq = eq / eq.iloc[0] * CAPITAL_REFERENCE_USD
    live = pd.Timestamp(LIVE_START_DATE)

    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=150)

    if equity_lev is not None and len(equity_lev.dropna()):
        lev = equity_lev.dropna()
        lev = lev / lev.iloc[0] * CAPITAL_REFERENCE_USD
        ax.plot(lev.index, lev.values, color="#cbd5e1", linewidth=1.0,
                label="Levier 1,5 (simulation, financement 2,5 %/an)", zorder=1)

    avant = eq.loc[eq.index < live]
    apres = eq.loc[eq.index >= live]
    if len(avant):
        ax.plot(avant.index, avant.values, color="#64748b", linewidth=1.3,
                label="Levier 1 — performance simulee (backtest)", zorder=2)
    if len(apres):
        # raccord visuel entre les deux segments
        if len(avant):
            join_x = [avant.index[-1]] + list(apres.index)
            join_y = [avant.values[-1]] + list(apres.values)
        else:
            join_x, join_y = apres.index, apres.values
        ax.plot(join_x, join_y, color="#0f766e", linewidth=1.6,
                label="Levier 1 — signaux live", zorder=3)

    if eq.index.min() <= live <= eq.index.max():
        ax.axvline(live, color="#b91c1c", linewidth=1.0, linestyle="--")
        ax.annotate("Demarrage live", xy=(live, ax.get_ylim()[1]),
                    xytext=(6, -14), textcoords="offset points",
                    fontsize=8, color="#b91c1c")

    ax.set_yscale("log")
    ax.set_ylabel(f"Capital ({DEVISE}, base {CAPITAL_REFERENCE_USD:,.0f})".replace(",", " "),
                  fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ==========================================================================
# EMAIL HTML
# ==========================================================================

ACTION_STYLES = {
    "OUVRIR":    {"color": "#15803d", "bg": "#dcfce7", "icon": "&#9650;"},
    "FERMER":    {"color": "#b91c1c", "bg": "#fee2e2", "icon": "&#9660;"},
    "MAINTENIR": {"color": "#475569", "bg": "#f1f5f9", "icon": "&#61;"},
}


def _alloc_rows(spy_pct, qqq_pct, cash_pct, capital):
    out = []
    data = [("SPY", "S&P 500", spy_pct, "#0f172a"),
            ("QQQ", "Nasdaq 100", qqq_pct, "#0f172a"),
            ("Liquidites", "Cash", cash_pct, "#64748b")]
    for i, (tk, name, pct, col) in enumerate(data):
        montant = capital * pct / 100
        bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        out.append(f"""
        <tr style="background:{bg};">
          <td style="padding:12px 16px;border-bottom:1px solid #e2e8f0;font-weight:700;color:{col};font-size:15px;">{tk}
            <span style="display:block;font-weight:400;color:#94a3b8;font-size:12px;margin-top:2px;">{name}</span></td>
          <td style="padding:12px 16px;border-bottom:1px solid #e2e8f0;text-align:right;font-weight:700;color:{col};font-size:18px;">{pct:.0f}%</td>
          <td style="padding:12px 16px;border-bottom:1px solid #e2e8f0;text-align:right;color:#475569;font-size:15px;">{montant:,.0f} $</td>
        </tr>""".replace(",", " "))
    return "".join(out)


def build_html_email(ctx, maxdd_1="n.d.", maxdd_15="n.d."):
    s = ACTION_STYLES[ctx["action"]]
    cap = ctx.get("capital_exemple", 10000)
    spy, qqq, cash = ctx["spy_pct"], ctx["qqq_pct"], ctx["cash_pct"]
    last = (f"""<p style="margin:4px 0 0;color:#94a3b8;font-size:13px;">Derniere action : {ctx['last_action']}</p>"""
            if ctx.get("last_action") else "")
    sub_lev = ""
    if cash < 99.9:
        sub_lev = ('<table role="presentation" width="100%" style="margin-top:10px;background:#fffbeb;'
                   'border:1px solid #fde68a;border-radius:8px;"><tr><td style="padding:10px 14px;color:#92400e;font-size:12px;line-height:1.5;">'
                   "<b>Levier :</b> allocation indiquee pour un levier de 1 (sans effet de levier). "
                   "Si vous utilisez un levier, multipliez chaque ligne par votre coefficient (ex. levier x1,5 : 50% deviennent 75%)."
                   "</td></tr></table>")
    alloc = _alloc_rows(spy, qqq, cash, cap)

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#eef2f6;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef2f6;padding:24px 12px;"><tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:14px;overflow:hidden;box-shadow:0 1px 3px rgba(15,23,42,.08);">
  <tr><td style="background:#0f172a;padding:22px 28px;">
    <table role="presentation" width="100%"><tr>
      <td style="color:#ffffff;font-size:18px;font-weight:800;letter-spacing:2px;">{BRAND.upper()}</td>
      <td align="right" style="color:#7dd3fc;font-size:12px;font-weight:600;">{STRAT_NAME}</td>
    </tr></table></td></tr>
  <tr><td style="padding:28px 28px 8px;">
    <span style="display:inline-block;background:{s['bg']};color:{s['color']};font-size:13px;font-weight:800;letter-spacing:1px;padding:7px 14px;border-radius:999px;">{s['icon']}&nbsp;&nbsp;{ctx['action']}</span>
    <h1 style="margin:16px 0 6px;font-size:24px;line-height:1.25;color:#0f172a;font-weight:800;">{ctx['directive']}</h1>
    <p style="margin:0;color:#64748b;font-size:14px;">Signal du {ctx['date']}</p>
    {last}</td></tr>
  <tr><td style="padding:18px 28px 4px;">
    <p style="margin:0 0 10px;color:#0f172a;font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;">Votre allocation cible</p>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;">
      <tr style="background:#0f172a;">
        <td style="padding:10px 16px;color:#cbd5e1;font-size:11px;font-weight:700;letter-spacing:.5px;">ACTIF</td>
        <td style="padding:10px 16px;color:#cbd5e1;font-size:11px;font-weight:700;text-align:right;">% CAPITAL</td>
        <td style="padding:10px 16px;color:#cbd5e1;font-size:11px;font-weight:700;text-align:right;">POUR {cap:,.0f} $ US</td>
      </tr>{alloc}
    </table>{sub_lev}</td></tr>
  <tr><td style="padding:18px 28px 4px;">
    <p style="margin:0 0 10px;color:#0f172a;font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;">Performance de la strategie</p>
    <img src="cid:equity_curve" alt="Courbe de performance" width="544" style="width:100%;max-width:544px;border:1px solid #e2e8f0;border-radius:10px;display:block;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;">
      <tr style="background:#0f172a;">
        <td style="padding:10px 16px;color:#cbd5e1;font-size:11px;font-weight:700;letter-spacing:.5px;">RISQUE HISTORIQUE</td>
        <td style="padding:10px 16px;color:#cbd5e1;font-size:11px;font-weight:700;text-align:right;white-space:nowrap;">LEVIER 1</td>
        <td style="padding:10px 16px;color:#cbd5e1;font-size:11px;font-weight:700;text-align:right;white-space:nowrap;">LEVIER 1,5</td>
      </tr>
      <tr style="background:#ffffff;">
        <td style="padding:12px 16px;font-size:13px;color:#0f172a;font-weight:700;">Pire baisse depuis un sommet</td>
        <td style="padding:12px 16px;text-align:right;font-weight:700;color:#0f172a;font-size:13px;white-space:nowrap;">{maxdd_1}</td>
        <td style="padding:12px 16px;text-align:right;font-weight:700;color:#0f172a;font-size:13px;white-space:nowrap;">{maxdd_15}</td>
      </tr>
    </table>
    <p style="margin:6px 0 0;color:#94a3b8;font-size:11px;line-height:1.5;">La pire baisse (drawdown maximum) est la plus forte perte temporaire que la strategie a connue avant de retrouver son niveau precedent.</p>
    <p style="margin:10px 0 0;color:#64748b;font-size:12px;line-height:1.6;"><b>Comment lire le graphique :</b> la courbe principale suit un capital de {cap:,.0f} $ US <b>sans effet de levier</b>, comme le tableau d'allocation ci-dessus. La courbe gris clair montre la meme strategie avec un levier de 1,5, c'est-a-dire en investissant une fois et demie le capital grace a un emprunt (cout de l'emprunt de 2,5 % par an deja deduit). Avant le trait rouge, il s'agit d'une <b>performance simulee</b> : ce que la strategie aurait fait dans le passe. Apres le trait rouge, la courbe suit les signaux reellement envoyes. Tous les frais de transaction sont inclus. Les performances passees ne prejugent pas des performances futures.</p>
  </td></tr>
  <tr><td style="padding:20px 28px 26px;border-top:1px solid #e2e8f0;">
    <p style="margin:0 0 10px;color:#334155;font-size:13px;line-height:1.7;"><b>Quand agir ?</b> Ce signal a ete calcule apres la cloture de la Bourse americaine du {ctx['date']}. Si une action est demandee ci-dessus, passez votre ordre <b>a l'ouverture de la prochaine seance de Wall Street</b> (15h30 heure de Paris), en choisissant un ordre "au marche". Si l'email indique qu'il n'y a rien a faire, vous n'avez aucun ordre a passer.</p>
    <p style="margin:0;color:#cbd5e1;font-size:10px;line-height:1.5;">Information fournie a titre d'aide a la decision &mdash; ne constitue pas un conseil en investissement personnalise. Les performances passees ne prejugent pas des performances futures. Vous restez responsable de vos ordres. &middot; <a href="#" style="color:#94a3b8;">Se desabonner</a></p>
  </td></tr>
</table></td></tr></table></body></html>"""


def subject_for(ctx):
    a = ctx["action"]
    if a == "OUVRIR":
        return f"{STRAT_NAME} - SIGNAL : ouvrir des positions ({ctx['date']})"
    if a == "FERMER":
        return f"{STRAT_NAME} - SIGNAL : sortir du marche ({ctx['date']})"
    return f"{STRAT_NAME} - Point hebdo : rien a faire ({ctx['date']})"


# ==========================================================================
# ENVOI AVEC REESSAIS
# ==========================================================================

def send_email(subject, html, sender, app_password, recipients, inline_png=None):
    last_err = None
    for attempt, delay in enumerate([0] + SMTP_RETRY_DELAYS):
        if delay:
            print(f"  Nouvelle tentative dans {delay} s ...")
            time.sleep(delay)
        try:
            msg = MIMEMultipart("related")
            msg["From"] = f"{BRAND} <{sender}>"
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            alt = MIMEMultipart("alternative")
            alt.attach(MIMEText(html, "html", "utf-8"))
            msg.attach(alt)
            if inline_png is not None:
                img = MIMEImage(inline_png, _subtype="png")
                img.add_header("Content-ID", "<equity_curve>")
                img.add_header("Content-Disposition", "inline",
                               filename="performance.png")
                msg.attach(img)
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=60) as srv:
                srv.starttls()
                srv.login(sender, app_password)
                srv.sendmail(sender, recipients, msg.as_string())
            print(f"  Email envoye a : {recipients} (tentative {attempt + 1})")
            return True
        except Exception as e:
            last_err = e
            print(f"  Echec tentative {attempt + 1} : {type(e).__name__} : {e}")
    raise RuntimeError(f"Envoi impossible apres {len(SMTP_RETRY_DELAYS) + 1} tentatives : {last_err}")


def send_tech_alert(message, sender, app_password):
    """Alerte technique interne. Jamais envoyee aux abonnes."""
    dest = os.environ.get("ALERT_RECIPIENTS_TECH", sender)
    recipients = [r.strip() for r in dest.split(",") if r.strip()]
    subject = f"[ALERTE TECHNIQUE] {STRAT_NAME} - envoi non effectue"
    html = (f"<html><body style='font-family:Arial,sans-serif;'>"
            f"<h2 style='color:#b91c1c;'>Signal non envoye</h2>"
            f"<p>Strategie : <b>{STRAT_NAME}</b></p>"
            f"<p>Horodatage : {datetime.now(NY).strftime('%Y-%m-%d %H:%M')} (New York)</p>"
            f"<pre style='background:#f1f5f9;padding:12px;border-radius:6px;"
            f"white-space:pre-wrap;'>{message}</pre></body></html>")
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{BRAND} <{sender}>"
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(html, "html", "utf-8"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=60) as srv:
            srv.starttls()
            srv.login(sender, app_password)
            srv.sendmail(sender, recipients, msg.as_string())
        print(f"  Alerte technique envoyee a : {recipients}")
    except Exception as e:
        print(f"  Alerte technique non envoyee : {e}")


# ==========================================================================
# JOURNAL DES SIGNAUX
# ==========================================================================

LOG_FIELDS = ["horodatage_envoi", "date_signal", "motif_signal", "action",
              "spy_pct", "qqq_pct", "cash_pct", "prix_spy", "prix_qqq",
              "source_donnees", "destinataires", "statut"]


def already_sent_today(date_signal: pd.Timestamp) -> bool:
    """Verifie dans le journal qu'aucun email n'est deja parti pour cette seance."""
    if not SIGNALS_LOG.exists():
        return False
    try:
        df = pd.read_csv(SIGNALS_LOG)
        if "date_signal" not in df.columns or df.empty:
            return False
        deja = df.loc[df["statut"] == "ENVOYE", "date_signal"].astype(str).tolist()
        return str(pd.Timestamp(date_signal).date()) in deja
    except Exception as e:
        print(f"  Journal illisible ({e}), on continue sans blocage.")
        return False


def append_log(ctx, motif, source, recipients, statut):
    is_new = not SIGNALS_LOG.exists()
    row = {
        "horodatage_envoi": datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S"),
        "date_signal": str(pd.Timestamp(ctx["signal_date"]).date()),
        "motif_signal": motif,
        "action": ctx["action"],
        "spy_pct": ctx["spy_pct"],
        "qqq_pct": ctx["qqq_pct"],
        "cash_pct": ctx["cash_pct"],
        "prix_spy": round(ctx["spy_close"], 4),
        "prix_qqq": round(ctx["qqq_close"], 4),
        "source_donnees": source,
        "destinataires": len(recipients),
        "statut": statut,
    }
    with open(SIGNALS_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    print(f"  Journal mis a jour : {SIGNALS_LOG}")


# ==========================================================================
# PROGRAMME PRINCIPAL
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Envoyer meme si ce n'est pas un jour de signal")
    ap.add_argument("--dry-run", action="store_true",
                    help="Tout calculer sans envoyer ni journaliser")
    args = ap.parse_args()

    sender = os.environ.get("GMAIL_USER")
    app_password = os.environ.get("GMAIL_APP_PASS")
    recipients = [r.strip() for r in os.environ.get("ALERT_RECIPIENTS", "").split(",") if r.strip()]

    if not sender or not app_password or not recipients:
        print("ERREUR : GMAIL_USER / GMAIL_APP_PASS / ALERT_RECIPIENTS manquant")
        sys.exit(1)

    # --- 1. Donnees --------------------------------------------------------
    try:
        print("Recuperation des donnees ...")
        close, open_adj, source = build_frames()
        print(f"  Source retenue : {source}")
        print(f"  Derniere seance recue : {close.index[-1].date()}")
    except Exception as e:
        msg = (f"Aucune source de donnees disponible.\n\n"
               f"{type(e).__name__} : {e}\n\n{traceback.format_exc()}")
        print(f"ERREUR : {msg}")
        send_tech_alert(msg, sender, app_password)
        sys.exit(1)

    # --- 2. Fraicheur ------------------------------------------------------
    fresh, fresh_msg = check_freshness(close)
    print(f"  {fresh_msg}")
    if not fresh and not args.force:
        send_tech_alert(
            f"Envoi annule : donnees non fraiches.\n\n{fresh_msg}\n\n"
            f"Source interrogee : {source}\n"
            f"Aucun email n'a ete envoye aux destinataires.",
            sender, app_password)
        sys.exit(1)

    # --- 3. Calcul de l'etat -----------------------------------------------
    try:
        ctx = compute_state(close, open_adj)
        position = compute_position_details(ctx["_eng"], close)
        signal_day, motif = is_signal_day(close)
        print(f"  Jour de signal : {signal_day} ({motif})")
        print(f"  Etat : {'INVESTI' if position['investi'] else 'LIQUIDITES'}"
              f" | gain latent {position['gain_latent']:+.2f} {DEVISE}"
              f" ({position['gain_latent_pct']:+.2f} %)")
    except Exception as e:
        msg = f"Calcul impossible.\n\n{type(e).__name__} : {e}\n\n{traceback.format_exc()}"
        print(f"ERREUR : {msg}")
        send_tech_alert(msg, sender, app_password)
        sys.exit(1)

    # --- 4. Ecriture de l'etat public (tous les jours ouvres) ---------------
    if args.dry_run:
        print("Mode test : etat non ecrit, aucun envoi.")
        print(f"  Action du jour : {ctx['action']} | {ctx['directive']}")
        return

    ecrire_etat_public(ctx, position, signal_day, motif, source)

    # --- 5. Envoi de l'email : uniquement les jours de signal ---------------
    if not signal_day and not args.force:
        print("Etat mis a jour. Aucun email : ce n'est pas un jour de signal.")
        return

    date_signal = close.index[-1]
    if already_sent_today(date_signal) and not args.force:
        print(f"Etat mis a jour. Aucun email : deja envoye pour la seance du {date_signal.date()}.")
        return

    try:
        print(f"  ACTION = {ctx['action']} | {ctx['directive']}")
        eng15 = run_modeC(close, open_adj, leverage=1.5)
        if eng15.get("margin_breaches", 0) > 0:
            print(f"  ATTENTION : {eng15['margin_breaches']} jour(s) sous le seuil "
                  f"de marge Reg-T 25% sur la simulation levier 1,5.")
        maxdd_1 = f"{max_drawdown_pct(ctx['_eng']['equity']):.1f} %".replace(".", ",")
        maxdd_15 = f"{max_drawdown_pct(eng15['equity']):.1f} %".replace(".", ",")
        html = build_html_email(ctx, maxdd_1=maxdd_1, maxdd_15=maxdd_15)
        png = build_equity_png(ctx["_eng"]["equity"], equity_lev=eng15["equity"])
        send_email(subject_for(ctx), html, sender, app_password, recipients,
                   inline_png=png)
        # Un envoi declenche a la main un jour ordinaire est marque comme test,
        # pour ne pas polluer la verification des signaux reellement diffuses.
        statut = "ENVOYE" if signal_day else "ENVOYE_TEST"
        append_log(ctx, motif, source, recipients, statut)
        print(f"Termine ({statut}).")
    except Exception as e:
        msg = f"{type(e).__name__} : {e}\n\n{traceback.format_exc()}"
        print(f"ERREUR : {msg}")
        try:
            append_log(ctx, motif, source, recipients, "ECHEC")
        except Exception:
            pass
        send_tech_alert(msg, sender, app_password)
        sys.exit(1)


if __name__ == "__main__":
    main()
