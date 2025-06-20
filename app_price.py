import streamlit as st
from curl_cffi import requests
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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

# Greeks (formule Black-Scholes)
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

# Choix du sous-jacent
ticker_symbol = st.text_input("Ticker Yahoo Finance (ex : AAPL, MSFT, ^FCHI ...)", "AAPL")
maturity = st.number_input("Maturité (en années)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)

with st.spinner("Chargement des données du marché..."):
    session = requests.Session(impersonate="chrome")
    ticker = yf.Ticker(ticker_symbol, session=session)
    spot = ticker.history(period="1d")["Close"].iloc[-1]
    hist = ticker.history(period="1y")["Close"]
    vol = hist.pct_change().std() * np.sqrt(252)
    try:
        div = ticker.dividends[-1]
    except:
        div = 0.0
    # Taux sans risque 10Y US
    riskfree_ticker = yf.Ticker("^TNX", session=session)
    rfree = riskfree_ticker.history(period="1d")["Close"].iloc[-1] / 100

col1, col2, col3 = st.columns(3)
col1.metric("Spot", f"{spot:.2f}")
col2.metric("Volatilité annualisée", f"{vol*100:.2f} %")
col3.metric("Taux sans risque (10Y)", f"{rfree*100:.2f} %")

st.markdown("---")

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
payoff_labels = []
fig, ax = plt.subplots(figsize=(10,5))

colors = ['b', 'g', 'r', 'm', 'y', 'c']

results = []

for idx in range(n_options):
    K = option_strikes[idx]
    type_ = option_types[idx]
    pos_ = option_pos[idx]
    qty = option_qty[idx]

    prime = black_scholes(spot, K, maturity, rfree, vol, type=type_)
    if type_ == 'call':
        payoff = (call_payoff(S_range, K) - prime) * qty
    else:
        payoff = (put_payoff(S_range, K) - prime) * qty
    if pos_ == 'short':
        payoff = -payoff
        prime_display = f"+{prime:.2f}"
        color_txt = "red"
    else:
        prime_display = f"-{prime:.2f}"
        color_txt = "green"

    payoff_total += payoff
    ax.plot(S_range, payoff, '--', label=f"{pos_} {type_} K={K} x{qty}", color=colors[idx % len(colors)])
    ax.annotate(f"Prime {prime_display}", xy=(K, 0), xytext=(K, np.max(payoff)*0.2),
                arrowprops=dict(arrowstyle="->", color=color_txt), color=color_txt, fontsize=9, ha='center')

    # Calcul des Greeks au spot
    d = delta(spot, K, maturity, rfree, vol, type=type_)
    g = gamma(spot, K, maturity, rfree, vol)
    v = vega(spot, K, maturity, rfree, vol)
    t = theta(spot, K, maturity, rfree, vol, type=type_)
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
        "Quantité": qty
    })

ax.plot(S_range, payoff_total, label="Payoff total", linewidth=2, color='black')
ax.axhline(0, color='black', linewidth=0.7)
ax.set_title("Payoff de la stratégie")
ax.set_xlabel("Spot à maturité")
ax.set_ylabel("Payoff (€)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.subheader("Tableau des résultats")
st.dataframe(results, hide_index=True)

# Optionnel : télécharger les résultats
import pandas as pd
df = pd.DataFrame(results)
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Télécharger les résultats (CSV)", csv, "resultats_options.csv", "text/csv")

