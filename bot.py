import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests

BUDGET = 10_000

UNIVERSE = [
    "AAPL","MSFT","JPM","KO","PG","XOM","JNJ","PEP","WMT","HD","UNH","V","MA","COST","ABBV","CRM","NFLX",
    "ADBE","DIS","MCD","NKE","INTC","CSCO","QCOM","TXN","AMGN","TMO","LIN","NEE","ORCL","BAC","WFC","IBM",
    "GE","CAT","DE","RTX","HON","SBUX","LOW","INTU","GS","MS","BLK","SCHW","C","CVX",
    "SPY","VOO","QQQ","VTI","SCHD","IWM","DIA"
]

def rsi14(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr14(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def higher_highs_higher_lows(data: pd.DataFrame, w: int = 20) -> bool:
    if len(data) < 2*w:
        return False
    last = data.iloc[-w:]
    prev = data.iloc[-2*w:-w]
    return (last["High"].max() > prev["High"].max()) and (last["Low"].min() > prev["Low"].min())

def safe_float(x):
    try:
        return float(x) if x is not None else np.nan
    except:
        return np.nan

def main():
    # Session pro obcházení blokování v GitHub Actions
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    rows = []

    for t in sorted(set(UNIVERSE)):
        try:
            # multi_level_index=False je klíčové pro správné načtení sloupců
            hist = yf.download(t, period="18mo", interval="1d", auto_adjust=True, 
                               progress=False, multi_level_index=False, session=session)
            
            if hist is None or hist.empty or len(hist) < 210:
                continue

            hist["SMA200"] = hist["Close"].rolling(200).mean()
            hist["EMA20"] = hist["Close"].ewm(span=20, adjust=False).mean()
            hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
            hist["RSI14"] = rsi14(hist["Close"], 14)
            hist["ATR14"] = atr14(hist["High"], hist["Low"], hist["Close"], 14)

            last = hist.iloc[-1]
            price = safe_float(last["Close"])
            sma200 = safe_float(last["SMA200"])
            rsi = safe_float(last["RSI14"])

            if np.isnan(price) or np.isnan(sma200) or np.isnan(rsi):
                continue

            # --- FILTRY (Mírně uvolněné pro reálný trh) ---
            if not (20 <= price <= 550): continue
            if price < sma200: continue
            if not (30 <= rsi <= 65): continue
            
            hhhl = higher_highs_higher_lows(hist, w=20)
            if not hhhl: continue

            # --- FUNDAMENTALS ---
            ticker_obj = yf.Ticker(t, session=session)
            info = ticker_obj.info or {}

            market_cap = safe_float(info.get("marketCap"))
            total_assets = safe_float(info.get("totalAssets"))
            avg_vol = safe_float(info.get("averageVolume")) or safe_float(info.get("averageVolume10days"))

            effective_cap = market_cap if not np.isnan(market_cap) else total_assets
            asset_type = "Stock" if not np.isnan(market_cap) else "ETF"

            if np.isnan(effective_cap) or effective_cap < 15_000_000_000: continue
            if np.isnan(avg_vol) or avg_vol < 800_000: continue

            # --- VÝPOČTY ---
            dist_sma200_pct = (price - sma200) / sma200 * 100
            atr = safe_float(last["ATR14"])
            swing_low = safe_float(hist.iloc[-20:]["Low"].min())
            
            stop = swing_low - (0.5 * atr if not np.isnan(atr) else 0)
            if stop >= price: stop = price * 0.97 # Pojistka

            risk = max(price - stop, 0.01)
            target = price + 2.5 * risk # Zvýšeno RR na 2.5
            
            name = info.get("shortName") or t
            rows.append({
                "Ticker": t,
                "Name": name,
                "AssetType": asset_type,
                "Price (USD)": round(price, 2),
                "MarketCap_or_TotalAssets": int(effective_cap),
                "RSI14": round(rsi, 2),
                "DistanceFromSMA200 %": round(dist_sma200_pct, 2),
                "Trend Confirmed": "Y" if (last["EMA20"] > last["EMA50"]) else "N",
                "Entry (USD)": round(price, 2),
                "StopLoss (USD)": round(stop, 2),
                "Target (USD)": round(target, 2),
                "Risk/Reward": round((target - price) / risk, 2),
            })

        except Exception as e:
            print(f"Error {t}: {e}")
            continue

    df = pd.DataFrame(rows)

    if not df.empty:
        # Seřazení a výběr top 5
        df["abs_dist"] = df["DistanceFromSMA200 %"].abs()
        df = df.sort_values(by=["Trend Confirmed", "abs_dist"], ascending=[False, True]).head(5).copy()
        
        alloc = BUDGET / len(df)
        df["Alloc $"] = round(alloc, 2)
        df["Shares"] = (alloc / df["Price (USD)"]).apply(math.floor)
        df.drop(columns=["abs_dist"], inplace=True)

    df.to_csv("candidates.csv", index=False)

if __name__ == "__main__":
    main()
