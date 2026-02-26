import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ----------------------------
# 1) UNIVERSE (blue-chip + ETF)
# ----------------------------
TICKERS = [
    # Blue-chip (příklady; můžeš rozšířit)
    "AAPL","MSFT","JPM","KO","PG","XOM","WMT","CVX","PEP","COST","MCD","HD","V","MA",
    "UNH","ABBV","MRK","BAC","CSCO","ORCL","ADBE","CRM","NFLX","DIS",
    # ETF
    "SPY","VOO","QQQ","VTI","DIA","IWM","SCHD"
]

# ----------------------------
# 2) Tvoje filtry
# ----------------------------
PRICE_MIN, PRICE_MAX = 30, 150
MCAP_MIN = 20_000_000_000      # 20B
AVG_VOL_MIN = 1_000_000        # 1M (30denní průměr)
RSI_MIN, RSI_MAX = 35, 55
MAX_CANDIDATES = 5

# (volitelné) Telegram
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def higher_highs_higher_lows(df: pd.DataFrame) -> bool:
    # jednoduchá, praktická definice: posledních 20 dní má vyšší high i vyšší low než předchozích 20 dní
    if len(df) < 60:
        return False
    h1 = df["High"].iloc[-20:].max()
    h0 = df["High"].iloc[-40:-20].max()
    l1 = df["Low"].iloc[-20:].min()
    l0 = df["Low"].iloc[-40:-20].min()
    return (h1 > h0) and (l1 > l0)

def send_telegram(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=20).raise_for_status()

def main():
    rows = []
    for tkr in TICKERS:
        try:
            t = yf.Ticker(tkr)

            # 1y data kvůli SMA200
            df = t.history(period="1y", interval="1d", auto_adjust=True)
            if df is None or df.empty or len(df) < 210:
                continue

            df["SMA200"] = df["Close"].rolling(200).mean()
            df["RSI14"] = rsi(df["Close"], 14)
            df["AvgVol30"] = df["Volume"].rolling(30).mean()

            last = df.iloc[-1]
            price = float(last["Close"])
            sma200 = float(last["SMA200"])
            rsi14 = float(last["RSI14"])
            avgvol = float(last["AvgVol30"])

            # market cap (Yahoo občas nevrátí; když nevrátí, ticker přeskočíme)
            info = {}
            try:
                info = t.fast_info
                mcap = info.get("market_cap", None)
            except Exception:
                mcap = None
            if mcap is None:
                try:
                    mcap = (t.info or {}).get("marketCap", None)
                except Exception:
                    mcap = None
            if mcap is None:
                continue

            # Filtry
            if not (PRICE_MIN <= price <= PRICE_MAX):
                continue
            if mcap < MCAP_MIN:
                continue
            if avgvol < AVG_VOL_MIN:
                continue
            if not (price > sma200):
                continue
            if not (RSI_MIN <= rsi14 <= RSI_MAX):
                continue
            if not higher_highs_higher_lows(df):
                continue

            dist = (price - sma200) / sma200 * 100

            # návrh vstupu/SL/Target (jednoduchý model – můžeš změnit)
            entry = round(price, 2)
            stop = round(price * 0.93, 2)     # -7%
            target = round(price * 1.14, 2)   # +14%
            rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else np.nan

            rows.append({
                "Ticker": tkr,
                "Price": round(price, 2),
                "MarketCap": int(mcap),
                "AvgVol30": int(avgvol),
                "RSI14": round(rsi14, 2),
                "SMA200": round(sma200, 2),
                "DistanceFromSMA200_%": round(dist, 2),
                "TrendConfirmed": "Y",
                "Entry": entry,
                "StopLoss": stop,
                "Target": target,
                "RiskReward": round(float(rr), 2) if not np.isnan(rr) else ""
            })

        except Exception as e:
            print(f"{tkr} error: {e}")

    out = pd.DataFrame(rows)
    if out.empty:
        msg = "✅ Screener doběhl: žádný ticker nesplnil filtry."
        send_telegram(msg)
        return

    # Seřazení: blíž k SMA200 + nižší RSI v okně
    out = out.sort_values(by=["DistanceFromSMA200_%", "RSI14"], ascending=[True, True]).head(MAX_CANDIDATES)

    out.to_csv("candidates.csv", index=False)

    lines = ["📈 TOP kandidáti (Growth+Stability):"]
    for _, r in out.iterrows():
        lines.append(f"- {r['Ticker']} | ${r['Price']} | RSI {r['RSI14']} | DistSMA200 {r['DistanceFromSMA200_%']}% | RR {r['RiskReward']}")
    send_telegram("\n".join(lines))

if __name__ == "__main__":
    main()
