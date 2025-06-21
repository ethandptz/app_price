import streamlit as st
from curl_cffi import requests
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import pandas as pd
from yahooquery import search as yq_search

# ---------------------------
# Fonctions Black-Scholes et Greeks
# ---------------------------
def black_scholes(S, K, T, r, sigma, type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

def call_payoff(S, K):
    return np.maximum(S - K, 0)
def put_payoff(S, K):
    return np.maximum(K - S, 0)

def delta(S, K, T, r, sigma, type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1
def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100
def theta(S, K, T, r, sigma, type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == 'call':
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

# ---------------------------
# Interface Streamlit
# ---------------------------

st.set_page_config(page_title="Option Pricer Visual", layout="wide")
st.title("🎯 Pricer & Visualiseur d'Options Européennes")

# --------- Recherche intelligente du sous-jacent ---------
st.subheader("Recherche de sous-jacent (par nom, secteur, ticker, etc.)")
keyword = st.text_input("Tape un mot-clé, un secteur ou un ticker (ex: 'oil', 'commodities', 'S&P500', 'apple', 'BZ=F', etc.)", "")

session = requests.Session(impersonate="chrome")
ticker_symbol = ""

if keyword.strip():
    # Obtenir 50 résultats max pour être vraiment exhaustif :
    results = yq_search(keyword, first_quote=0, quotes_count=20)
    if results and "quotes" in results:
        tickers = [
            f"{item.get('symbol', '')} — {item.get('shortname', '') or item.get('longname', '') or item.get('name', '') or ''}"
            for item in results["quotes"]
            if "symbol" in item
        ]
        if tickers:
            selected = st.selectbox("Choisis un sous-jacent :", tickers, key="sousjacent")
            ticker_symbol = selected.split(' — ')[0]
            ticker_label = selected
        else:
            st.warning("Aucun ticker trouvé pour ce mot-clé. Utilise un code Yahoo Finance.")
            ticker_label = ""
    else:
        st.warning("Aucun résultat pour ce mot-clé. Utilise un code Yahoo Finance.")
        ticker_label = ""
else:
    ticker_symbol = st.text_input("Ticker Yahoo Finance (ex : AAPL, MSFT, ^FCHI ...)", "AAPL")
    ticker_label = ticker_symbol

# Afficher le ticker sélectionné
if not ticker_symbol:
    st.stop()

# --------- Choix du taux sans risque ---------
st.subheader("Choix du taux sans risque")

riskfree_dict = {
    "US 10Y (OAT)": "^TNX",
    "France 10Y (OAT)": "^FR10Y-GT",
    "Germany 10Y (Bund)": "^DE10Y-GT",
    "LIBOR USD 3M": "^USD3MTD156N",
    "EURIBOR 3M": "EURIBOR3MD.EU"
}
selected_rf = st.selectbox("Sélectionne le taux sans risque :", list(riskfree_dict.keys()))
rf_ticker = riskfree_dict[selected_rf]

with st.spinner("Chargement des données du marché..."):
    ticker = yf.Ticker(ticker_symbol, session=session)
    spot = ticker.history(period="1d")["Close"].iloc[-1]
    hist = ticker.history(period="1y")["Close"]
    vol = hist.pct_change().std() * np.sqrt(252)
    try:
        div = ticker.dividends[-1]
    except:
        div = 0.0

    # Taux sans risque sélectionné
    rf_data = yf.Ticker(rf_ticker, session=session)
    try:
        rfree = rf_data.history(period="1d")["Close"].iloc[-1] / 100
    except:
        rfree = 0.01

col1, col2, col3 = st.columns(3)
col1.metric("Spot", f"{spot:.2f}")
col2.metric("Volatilité annualisée", f"{vol*100:.2f} %")
col3.metric(selected_rf, f"{rfree*100:.2f} %")

st.markdown("---")

# --------- Sélecteur de date d'échéance ---------
today = datetime.today().date()
st.markdown("#### Choix de l'échéance")
maturity_date = st.date_input(
    "Date d’échéance (option européenne)",
    value=today + timedelta(days=365),
    min_value=today + timedelta(days=1)
)
days_to_maturity = (maturity_date - today).days
T = max(days_to_maturity / 365, 1/365)  # Jamais zéro pour la division

st.info(f"Nombre de jours jusqu'à échéance : **{days_to_maturity}** (soit {T:.2f} années)")

st.markdown("---")

# --------- Sélecteur du Greek à afficher ---------
st.markdown("#### Visualiser une courbe :")
greek_to_plot = st.selectbox(
    "Sélectionne ce que tu veux afficher :",
    ["Payoff", "Delta", "Gamma", "Vega", "Theta"]
)

# Table input pour les options du portefeuille
st.subheader("Configuration de la stratégie")
n_options = st.number_input("Nombre d'options dans la stratégie", min_value=1, max_value=5, value=1, step=1)
option_types = []
option_pos = []
option_strikes = []
option_qty = []

for i in range(n_options):
    cols = st.columns(4)
    type_ = cols[0].selectbox(f"Type option {i+1}", ["call", "put"], key=f"type_{i}")
    pos_ = cols[1].selectbox(f"Position {i+1}", ["long", "short"], key=f"pos_{i}")
    K_ = cols[2].number_input(f"Strike {i+1}", min_value=0.0, value=float(round(spot)), key=f"K_{i}")
    qty_ = cols[3].number_input(f"Quantité {i+1}", min_value=1, value=1, key=f"qty_{i}")
    option_types.append(type_)
    option_pos.append(pos_)
    option_strikes.append(K_)
    option_qty.append(qty_)

# Calculs & Affichage
S_range = np.linspace(spot * 0.5, spot * 1.5, 200)
payoff_total = np.zeros_like(S_range)
greek_total = np.zeros_like(S_range)
colors = ['b', 'g', 'r', 'm', 'y', 'c']
results = []

fig, ax = plt.subplots(figsize=(10,5))

for idx in range(n_options):
    K = option_strikes[idx]
    type_ = option_types[idx]
    pos_ = option_pos[idx]
    qty = option_qty[idx]

    prime = black_scholes(spot, K, T, rfree, vol, type=type_)
    
    # Pour chaque option, calcul du Greek ou du payoff sur toute la grille
    if greek_to_plot == "Payoff":
        if type_ == 'call':
            payoff = (call_payoff(S_range, K) - prime) * qty
        else:
            payoff = (put_payoff(S_range, K) - prime) * qty
        if pos_ == 'short':
            payoff = -payoff
        payoff_total += payoff
        ax.plot(S_range, payoff, '--', label=f"{pos_} {type_} K={K} x{qty}", color=colors[idx % len(colors)])
    else:
        if greek_to_plot == "Delta":
            greek = delta(S_range, K, T, rfree, vol, type=type_)
        elif greek_to_plot == "Gamma":
            greek = gamma(S_range, K, T, rfree, vol)
        elif greek_to_plot == "Vega":
            greek = vega(S_range, K, T, rfree, vol)
        elif greek_to_plot == "Theta":
            greek = theta(S_range, K, T, rfree, vol, type=type_)
        if pos_ == 'short':
            greek = -greek * qty
        else:
            greek = greek * qty
        greek_total += greek
        ax.plot(S_range, greek, '--', label=f"{pos_} {type_} K={K} x{qty}", color=colors[idx % len(colors)])

# Tracer la courbe globale (stratégie) en noir
if greek_to_plot == "Payoff":
    ax.plot(S_range, payoff_total, label="Payoff total", linewidth=2, color='black')
    ax.set_ylabel("Payoff (€)")
else:
    ax.plot(S_range, greek_total, label=f"{greek_to_plot} total", linewidth=2, color='black')
    ax.set_ylabel(greek_to_plot)

ax.axhline(0, color='black', linewidth=0.7)
ax.set_title(f"{greek_to_plot} de la stratégie")
ax.set_xlabel("Spot à maturité")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.subheader("Tableau des résultats")

# Calculs des greeks au spot pour le tableau
for idx in range(n_options):
    K = option_strikes[idx]
    type_ = option_types[idx]
    pos_ = option_pos[idx]
    qty = option_qty[idx]

    prime = black_scholes(spot, K, T, rfree, vol, type=type_)
    d = delta(spot, K, T, rfree, vol, type=type_)
    g = gamma(spot, K, T, rfree, vol)
    v = vega(spot, K, T, rfree, vol)
    t = theta(spot, K, T, rfree, vol, type=type_)
    if pos_ == 'short':
        d, g, v, t = -d*qty, -g*qty, -v*qty, -t*qty
    else:
        d, g, v, t = d*qty, g*qty, v*qty, t*qty

    results.append({
        "Type": type_,
        "Position": pos_,
        "Strike": K,
        "Prime": round(prime, 2),
        "Delta": round(d, 4),
        "Gamma": round(g, 4),
        "Vega": round(v, 4),
        "Theta": round(t, 4),
        "Quantité": qty,
        "Jours jusqu'à échéance": days_to_maturity
    })

st.dataframe(results, hide_index=True)

# Optionnel : télécharger les résultats
df = pd.DataFrame(results)
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Télécharger les résultats (CSV)", csv, "resultats_options.csv", "text/csv")
