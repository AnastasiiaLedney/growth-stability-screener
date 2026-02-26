import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# Nastavení
BUDGET = 10_000
UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","MA",
    "PG","KO","PEP","COST","WMT","XOM","CVX","JNJ","ABBV","MRK",
    "SPY","VOO","QQQ","DIA","SCHD","IWM"
]

def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def main():
    # 1. Ošetření blokování (User-Agent)
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    
    results = []

    print(f"Spouštím skener pro {len(UNIVERSE)} titulů...")

    for ticker in UNIVERSE:
        try:
            # 2. KLÍČOVÁ OPRAVA: multi_level_index=False
            df = yf.download(ticker, period="18mo", interval="1d", 
                             auto_adjust=True, progress=False, 
                             multi_level_index=False, session=session)
            
            if df.empty or len(df) < 201:
                continue

            # Výpočty
            close_prices = df["Close"]
            current_price = float(close_prices.iloc[-1])
            sma200 = close_prices.rolling(200).mean().iloc[-1]
            rsi = get_rsi(close_prices).iloc[-1]
            
            # 3. FILTRY (nastaveny tak, aby v roce 2026 něco našly)
            # Cena 20 - 800 USD
            if not (20 <= current_price <= 800):
                continue
            
            # Musí být nad SMA200 (rostoucí trend)
            if current_price < sma200:
                continue
            
            # RSI 30 - 70 (zdravý trend, ne extrémně překoupeno)
            if not (30 <= rsi <= 70):
                continue

            # Pokud prošel filtry, přidáme data
            results.append({
                "Ticker": ticker,
                "Price": round(current_price, 2),
                "SMA200": round(sma200, 2),
                "RSI": round(rsi, 2),
                "Status": "Koupit"
            })
            print(f"✅ {ticker} vyhovuje.")

        except Exception as e:
            print(f"❌ Chyba u {ticker}: {e}")

    # 4. Uložení výsledků
    final_df = pd.DataFrame(results)
    
    if final_df.empty:
        final_df = pd.DataFrame([{"Zpráva": "Dnes žádná akcie nesplnila filtry (zkuste uvolnit RSI nebo cenu)."}])
        print("Nebyla nalezena žádná vhodná akcie.")
    else:
        # Výpočet alokace
        num_stocks = len(final_df)
        alloc = BUDGET / num_stocks
        final_df["Alloc_USD"] = round(alloc, 2)
        final_df["Shares"] = (alloc / final_df["Price"]).apply(math.floor)

    final_df.to_csv("candidates.csv", index=False)
    print("Soubor candidates.csv byl vytvořen.")

if __name__ == "__main__":
    main()
