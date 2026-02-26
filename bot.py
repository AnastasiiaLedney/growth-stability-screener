import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests

BUDGET = 10_000
UNIVERSE = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","V","MA",
            "PG","KO","PEP","COST","WMT","XOM","CVX","JNJ","ABBV","MRK",
            "SPY","VOO","QQQ","DIA"]

def main():
    # Session proti blokování
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/110.0.0.0'})
    
    rows = []
    print(f"Skenuji {len(UNIVERSE)} titulů...")

    for t in UNIVERSE:
        try:
            # multi_level_index=False zajistí, že hist['Close'] bude fungovat
            hist = yf.download(t, period="18mo", interval="1d", auto_adjust=True, 
                               progress=False, multi_level_index=False, session=session)
            
            if hist.empty or len(hist) < 200: continue

            price = float(hist["Close"].iloc[-1])
            sma200 = hist["Close"].rolling(200).mean().iloc[-1]
            
            # --- UVOLNĚNÉ FILTRY ---
            # 1. Cena: do 1000 USD (aby prošel i Microsoft/Apple/Costco)
            if not (10 <= price <= 1000): continue
            
            # 2. Nad SMA200: Musí být v trendu
            if price < sma200: continue
            
            # 3. RSI: Rozšířeno na 30-75 (aby prošly i rostoucí akcie)
            delta = hist["Close"].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
            if not (30 <= rsi <= 75): continue

            # Pokud projde, přidáme do seznamu
            rows.append({
                "Ticker": t,
                "Cena": round(price, 2),
                "RSI": round(rsi, 1),
                "Nad_SMA200": "Ano"
            })
            print(f"Nalezeno: {t}")

        except Exception as e:
            print(f"Chyba u {t}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        # Pokud je stále prázdno, vytvoříme aspoň info řádek
        df = pd.DataFrame([{"Zpráva": "Žádná akcie nesplnila filtry"}])
    
    df.to_csv("candidates.csv", index=False)
    print("Hotovo. Výsledky v candidates.csv")

if __name__ == "__main__":
    main()
