"""
alert_email.py — Alerte email US Index Momentum (Quant Signals).

- Fetch SPY/QQQ/BIL + IRX en direct depuis EODHD (autonome, rien a synchroniser).
- Rejoue la strategie (mode C : signal close J, ordre a l'ouverture J+1) pour
  determiner l'ACTION du jour : OUVRIR / FERMER / MAINTENIR. Calcul a partir des
  donnees a chaque run -> aucun fichier d'etat a gerer.
- Construit un email HTML soigne (allocation a levier 1, note pour appliquer son
  propre levier) et l'envoie via Gmail.

Cadence (pilotee par le workflow GitHub) : chaque mercredi + dernier ouvre du mois.
Email envoye meme quand il n'y a rien a faire (MAINTENIR).

Dependances : pandas, numpy, requests + strategy.py (meme repo).
Variables d'environnement : GMAIL_USER, GMAIL_APP_PASS, ALERT_RECIPIENTS.
"""
import os
import sys
import smtplib
import traceback
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import requests

# Rend l'import de strategy.py robuste : racine du repo, src/, ou meme dossier
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "src"), os.path.join(_here, ".."), os.path.join(_here, "..", "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.strategy import (
        ASSETS_HELD, COST_BPS, CASH_YIELD_DAILY, COST_FINANCING_ANNUAL,
        compute_monthly_score, compute_target_allocation,
        get_filter_days, evaluate_filter, get_current_signal,
    )
except ImportError:
    from strategy import (
        ASSETS_HELD, COST_BPS, CASH_YIELD_DAILY, COST_FINANCING_ANNUAL,
        compute_monthly_score, compute_target_allocation,
        get_filter_days, evaluate_filter, get_current_signal,
    )

EODHD_KEY = os.environ.get("EODHD_KEY", "69fdc152a61830.85937256")
SMTP_SERVER, SMTP_PORT = "smtp.gmail.com", 587
CAPITAL_EXEMPLE = 10000
BRAND = "Quant Signals"
STRAT_NAME = "US Index Momentum"


# ====================== DONNEES (EODHD direct) ======================
def _eodhd(symbol, frm="1999-01-01"):
    url = f"https://eodhd.com/api/eod/{symbol}"
    r = requests.get(url, params={"api_token": EODHD_KEY, "fmt": "json",
                                  "from": frm, "period": "d"}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError(f"EODHD: aucune donnee pour {symbol}")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def build_frames():
    """close (TR : SPY/QQQ adjusted, BIL etendu via IRX) + open ajuste."""
    spy, qqq = _eodhd("SPY.US"), _eodhd("QQQ.US")
    bil, irx = _eodhd("BIL.US"), _eodhd("IRX.INDX")

    close = pd.DataFrame({"SPY": spy["adjusted_close"], "QQQ": qqq["adjusted_close"]})
    open_adj = pd.DataFrame({
        "SPY": spy["open"] * (spy["adjusted_close"] / spy["close"]),
        "QQQ": qqq["open"] * (qqq["adjusted_close"] / qqq["close"]),
    })
    # BIL : reel + extension IRX pre-2007
    bil_real = bil["adjusted_close"].copy()
    splice = bil_real.index.min()
    irx_pre = irx.loc[irx.index < splice, "close"]
    daily_ret = (1 + irx_pre / 100.0) ** (1 / 252) - 1
    synth = (1 + daily_ret).cumprod()
    synth = synth * (bil_real.iloc[0] / synth.iloc[-1])
    bil_full = pd.concat([synth, bil_real]).sort_index()
    bil_full = bil_full[~bil_full.index.duplicated(keep="last")]
    close["BIL"] = bil_full.reindex(close.index).ffill()

    close = close.sort_index().ffill().dropna()
    open_adj = open_adj.reindex(close.index)
    return close, open_adj


# ====================== MOTEUR (mode C, levier 1) ======================
def run_modeC(prices, prices_open, leverage=1.0, initial_capital=10_000_000):
    """Rejoue la strategie en mode C. Renvoie positions finales, ordres en attente
    (= action du jour, a executer a la prochaine ouverture), trades, equity."""
    prices_eom = prices.resample("ME").last()
    score = compute_monthly_score(prices_eom)
    alloc_target = compute_target_allocation(score).loc[score.dropna().index]

    rebal_dates, valid_idx = [], []
    for d in alloc_target.index:
        cands = prices.index[prices.index <= d]
        if len(cands):
            mapped = cands[-1]
            if mapped.month == d.month and (d - mapped).days <= 4:
                rebal_dates.append(mapped); valid_idx.append(d)
    alloc_bt = alloc_target.loc[valid_idx].copy()
    alloc_bt.index = pd.DatetimeIndex(rebal_dates)
    alloc_bt = alloc_bt[~alloc_bt.index.duplicated(keep="last")]
    prices_bt = prices.loc[prices.index >= alloc_bt.index.min()].copy()

    cash = float(initial_capital)
    positions = {a: 0.0 for a in ASSETS_HELD}
    trades, pending = [], []
    rebal_set = set(alloc_bt.index)
    prev_target = None
    filter_days = get_filter_days(prices_bt.index)
    forced_cash = False

    for date in prices_bt.index:
        p = prices_bt.loc[date]
        # exec des ordres en attente a l'ouverture
        if pending:
            try:
                po = prices_open.loc[date]
                for o in pending:
                    a, q = o["asset"], o["qty"]; px = float(po[a]); val = q * px; c = val * COST_BPS
                    if o["side"] == "BUY":
                        cash -= val + c; positions[a] += q
                    else:
                        cash += val - c; positions[a] -= q
                    trades.append({"date": date, "asset": a, "side": o["side"],
                                   "qty": q, "price": px, "reason": o["reason"]})
                pending = []
            except KeyError:
                pass
        if cash > 0:
            cash *= 1 + CASH_YIELD_DAILY

        # rebalancement mensuel
        if date in rebal_set:
            target = alloc_bt.loc[date].to_dict()
            changed = (prev_target is None) or any(abs(target[k] - prev_target.get(k, 0)) > 1e-9 for k in target)
            empty = sum(positions.values()) < 1e-6 and any(target.get(a, 0) > 0 for a in ASSETS_HELD)
            if changed or empty:
                eq = cash + sum(positions[a] * p[a] for a in ASSETS_HELD)
                tq = {a: int(eq * target.get(a, 0) * leverage / p[a]) if target.get(a, 0) > 0 else 0 for a in ASSETS_HELD}
                for a in ASSETS_HELD:
                    dq = tq[a] - positions[a]
                    if dq < 0:
                        pending.append({"asset": a, "side": "SELL", "qty": abs(dq), "reason": "ARBITRAGE"})
                    elif dq > 0:
                        pending.append({"asset": a, "side": "BUY", "qty": dq, "reason": "ARBITRAGE"})
                prev_target = target.copy(); forced_cash = False

        # filtre 15j (mercredi)
        if date in filter_days and not forced_cash:
            if evaluate_filter(prices_bt, date, positions):
                for a in ASSETS_HELD:
                    if positions[a] > 0:
                        pending.append({"asset": a, "side": "SELL", "qty": positions[a], "reason": "FORCE_EXIT"})
                forced_cash = True

    last_trade = trades[-1] if trades else None
    return {"positions": positions, "pending": pending, "trades": trades,
            "last_trade": last_trade, "last_date": prices_bt.index[-1]}


# ====================== ETAT -> CONTEXTE EMAIL ======================
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

    # classification action
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

    # date de declenchement du filtre (= date de la sortie FORCE_EXIT)
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
        ctx_market=dict(
            mom_ok=(monthly == "RISKY"),
            mom_txt=("Favorable" if monthly == "RISKY" else "Defavorable") + (" (neutralise par le filtre)" if (monthly == "RISKY" and filt) else ""),
            filter_on=filt,
            filter_txt=filter_txt,
            spy=f"{spy_c:,.2f}".replace(",", " "), qqq=f"{qqq_c:,.2f}".replace(",", " "),
        ),
    )


# ====================== EMAIL HTML (design) ======================
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
        eur = capital * pct / 100
        bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        out.append(f"""
        <tr style="background:{bg};">
          <td style="padding:12px 16px;border-bottom:1px solid #e2e8f0;font-weight:700;color:{col};font-size:15px;">{tk}
            <span style="display:block;font-weight:400;color:#94a3b8;font-size:12px;margin-top:2px;">{name}</span></td>
          <td style="padding:12px 16px;border-bottom:1px solid #e2e8f0;text-align:right;font-weight:700;color:{col};font-size:18px;">{pct:.0f}%</td>
          <td style="padding:12px 16px;border-bottom:1px solid #e2e8f0;text-align:right;color:#475569;font-size:15px;">{eur:,.0f} &euro;</td>
        </tr>""".replace(",", " "))
    return "".join(out)


def build_html_email(ctx):
    s = ACTION_STYLES[ctx["action"]]
    cap = ctx.get("capital_exemple", 10000)
    spy, qqq, cash = ctx["spy_pct"], ctx["qqq_pct"], ctx["cash_pct"]
    m = ctx.get("ctx_market", {})
    last = f"""<p style="margin:4px 0 0;color:#94a3b8;font-size:13px;">Derniere action : {ctx['last_action']}</p>""" if ctx.get("last_action") else ""
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
        <td style="padding:10px 16px;color:#cbd5e1;font-size:11px;font-weight:700;text-align:right;">POUR {cap:,.0f} &euro;</td>
      </tr>{alloc}
    </table>{sub_lev}</td></tr>
  <tr><td style="padding:20px 28px 26px;border-top:1px solid #e2e8f0;">
    <p style="margin:0 0 8px;color:#94a3b8;font-size:11px;line-height:1.5;">Signal genere automatiquement le {ctx['date']}. Strategie {STRAT_NAME} ({BRAND}). Les ordres sont a executer a l'ouverture de la prochaine seance.</p>
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


# ====================== ENVOI ======================
def send_email(subject, html, sender, app_password, recipients):
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{BRAND} <{sender}>"
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(html, "html", "utf-8"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as srv:
        srv.starttls(); srv.login(sender, app_password)
        srv.sendmail(sender, recipients, msg.as_string())
    print(f"Email envoye a : {recipients}")


def main():
    sender = os.environ.get("GMAIL_USER")
    app_password = os.environ.get("GMAIL_APP_PASS")
    recipients = [r.strip() for r in os.environ.get("ALERT_RECIPIENTS", "").split(",") if r.strip()]
    if not sender or not app_password or not recipients:
        print("ERREUR: GMAIL_USER / GMAIL_APP_PASS / ALERT_RECIPIENTS manquant"); sys.exit(1)
    try:
        print("Fetch EODHD..."); close, open_adj = build_frames()
        print(f"Derniere donnee : {close.index[-1].date()}")
        ctx = compute_state(close, open_adj)
        print(f"ACTION = {ctx['action']} | {ctx['directive']}")
        html = build_html_email(ctx)
        send_email(subject_for(ctx), html, sender, app_password, recipients)
        print("OK")
    except Exception as e:
        print(f"ERREUR: {type(e).__name__}: {e}\n{traceback.format_exc()}"); sys.exit(1)


if __name__ == "__main__":
    main()
