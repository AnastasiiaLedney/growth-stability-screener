import math
import numpy as np
import pandas as pd
import yfinance as yf

BUDGET = 10_000

# “Blue-chip” + pár likvidních ETF (můžeš kdykoli rozšířit)
UNIVERSE = [
    # Blue-chips (příklady)
    "AAPL","MSFT","JPM","KO","PG","XOM","JNJ","PEP","WMT","HD","UNH","V","MA","COST","ABBV","CRM","NFLX",
    "ADBE","DIS","MCD","NKE","INTC","CSCO","QCOM","TXN","AMGN","TMO","LIN","NEE","ORCL","BAC","WFC","IBM",
    "GE","CAT","DE","RTX","HON","SBUX","LOW","INTU","GS","MS","BLK","SCHW","C","CVX",
    # ETF
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
    # jednoduchá, ale použitelná definice:
    # max posledních w dnů > max předchozích w dnů
    # min posledních w dnů > min předchozích w dnů
    if len(data) < 2*w:
        return False
    last = data.iloc[-w:]
    prev = data.iloc[-2*w:-w]
    return (last["High"].max() > prev["High"].max()) and (last["Low"].min() > prev["Low"].min())

def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def main():
    rows = []

    for t in sorted(set(UNIVERSE)):
        try:
            hist = yf.download(t, period="18mo", interval="1d", auto_adjust=True, progress=False)
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

            # --- filtr price ---
            if not (30 <= price <= 150):
                continue

            # --- základní “trend” filtry ---
            above_sma200 = price > sma200
            trend_confirmed = above_sma200 and (last["EMA20"] > last["EMA50"])

            if not above_sma200:
                continue

            # --- RSI filtr ---
            if not (35 <= rsi <= 55):
                continue

            # --- HH/HL filtr ---
            hhhl = higher_highs_higher_lows(hist, w=20)
            if not hhhl:
                continue

            # --- fundamentals / liquidity ---
            info = {}
            try:
                info = yf.Ticker(t).info or {}
            except Exception:
                info = {}

            market_cap = safe_float(info.get("marketCap"))
            total_assets = safe_float(info.get("totalAssets"))  # ETF proxy
            avg_vol = safe_float(info.get("averageVolume")) or safe_float(info.get("averageVolume10days"))

            # Market cap: akcie marketCap, ETF totalAssets
            # (u ETF je “market cap” často prázdný)
            effective_cap = market_cap
            asset_type = "Stock"
            if np.isnan(effective_cap) and not np.isnan(total_assets):
                effective_cap = total_assets
                asset_type = "ETF"

            if np.isnan(effective_cap) or effective_cap < 20_000_000_000:
                continue

            if np.isnan(avg_vol) or avg_vol < 1_000_000:
                continue

            # --- výpočty pro tabulku ---
            dist_sma200_pct = (price - sma200) / sma200 * 100

            # entry = aktuální cena (můžeš změnit na limit pod close)
            entry = price

            # stop-loss: swing low za posledních 20 dnů mínus “polštář” 0.5*ATR
            last20 = hist.iloc[-20:]
            atr = safe_float(last["ATR14"])
            swing_low = safe_float(last20["Low"].min())
            stop = swing_low - (0.5 * atr if not np.isnan(atr) else 0)

            # target: RR 2:1 (konzervativní default)
            risk = max(entry - stop, 0.0001)
            target = entry + 2 * risk
            rr = (target - entry) / risk

            # zatím vybereme “top” později řazením
            name = info.get("shortName") or info.get("longName") or ""
            rows.append({
                "Ticker": t,
                "Name": name,
                "AssetType": asset_type,
                "Price (USD)": round(price, 2),
                "MarketCap_or_TotalAssets (USD)": int(effective_cap),
                "Avg Daily Volume": int(avg_vol),
                "RSI14": round(rsi, 2),
                "SMA200": round(sma200, 2),
                "DistanceFromSMA200 %": round(dist_sma200_pct, 2),
                "HigherHighsHigherLows": hhhl,
                "Trend Confirmed (Y/N)": "Y" if trend_confirmed else "N",
                "Entry (USD)": round(entry, 2),
                "StopLoss (USD)": round(stop, 2),
                "Target (USD)": round(target, 2),
                "Risk/Reward": round(rr, 2),
            })

        except Exception:
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        # i když nic nenajde, vytvoří soubor, aby workflow nespadlo
        df = pd.DataFrame(columns=[
            "Ticker","Name","AssetType","Price (USD)","MarketCap_or_TotalAssets (USD)","Avg Daily Volume",
            "RSI14","SMA200","DistanceFromSMA200 %","HigherHighsHigherLows","Trend Confirmed (Y/N)",
            "Entry (USD)","StopLoss (USD)","Target (USD)","Risk/Reward","Alloc $ (of 10k)","Shares (est.)","Notes"
        ])
        df.to_csv("candidates.csv", index=False)
        return

    # --- vyber maximálně 5 kandidátů ---
    # řazení: nejdřív “trend confirmed”, pak nejblíž SMA200 (pullback), pak největší cap
    df["trend_rank"] = (df["Trend Confirmed (Y/N)"] == "Y").astype(int)
    df["cap_rank"] = df["MarketCap_or_TotalAssets (USD)"]
    df["abs_dist"] = df["DistanceFromSMA200 %"].abs()

    df = df.sort_values(
        by=["trend_rank","abs_dist","cap_rank"],
        ascending=[False, True, False]
    ).head(5).copy()

    # --- alokace a počty kusů ---
    n = len(df)
    alloc_each = BUDGET / n if n else 0

    df["Alloc $ (of 10k)"] = (alloc_each).round(2)
    df["Shares (est.)"] = df.apply(lambda r: int(math.floor(alloc_each / r["Price (USD)"])) if r["Price (USD)"] > 0 else 0, axis=1)

    df["Notes"] = "Meets: 30–150$, cap/asset≥20B, vol≥1M, >SMA200, RSI 35–55, HH&HL"

    df = df.drop(columns=["trend_rank","cap_rank","abs_dist"], errors="ignore")

    df.to_csv("candidates.csv", index=False)

if __name__ == "__main__":
    main()
