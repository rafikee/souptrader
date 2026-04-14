import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
import datetime as dt
import pandas as pd
from finvizfinance.screener.overview import Overview
import json
import pytz
import os
from aatinaa import sharia_status
import time
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

load_dotenv()

APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

alpaca_client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)

# Finviz filters: core set (same as v1) and expanded CANSLIM additions
CORE_FILTERS = {
    "Market Cap.": "+Mid (over $2bln)",
    "Average Volume": "Over 1M",
    "200-Day Simple Moving Average": "Price above SMA200",
    "50-Day Simple Moving Average": "Price above SMA50",
    "52-Week High/Low": "30% or more above Low",
    "EPS growthqtr over qtr": "Over 20%",
    "Sales growthqtr over qtr": "Over 20%",
}

EXPANDED_FILTERS = {
    **CORE_FILTERS,
    "EPS growthpast 5 years": "Over 20%",       # (A) Annual earnings quality
    "Return on Equity": "Over +15%",              # Profitability
    "InstitutionalOwnership": "Over 50%",         # (I) Smart money confirmation
    "InsiderTransactions": "Positive (>0%)",      # Insiders buying
}


def get_stock_data(ticker, start_date, end_date, max_retries=3):
    """Fetch daily bars from Alpaca. Returns DataFrame with OHLCV or None."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(0.1)

            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
            )
            bars = alpaca_client.get_stock_bars(request)

            if not bars or len(bars.data) == 0:
                return None

            raw_data = bars.data
            if not (isinstance(raw_data, dict) and ticker in raw_data):
                return None

            data_list = []
            for item in raw_data[ticker]:
                if hasattr(item, "timestamp") and hasattr(item, "close"):
                    data_list.append({
                        "timestamp": item.timestamp,
                        "Open": item.open,
                        "High": item.high,
                        "Low": item.low,
                        "Close": item.close,
                        "Volume": item.volume,
                    })
                elif isinstance(item, dict):
                    data_list.append(item)

            if not data_list:
                return None

            df = pd.DataFrame(data_list)
            if "timestamp" in df.columns:
                df["Date"] = pd.to_datetime(df["timestamp"])
                df.set_index("Date", inplace=True)
                df.drop("timestamp", axis=1, inplace=True)

            if "Close" not in df.columns or len(df) < 50:
                return None

            return df

        except Exception as e:
            msg = str(e)
            if "not found" in msg.lower() or "404" in msg:
                return None
            wait = 1 * (attempt + 1) if ("rate limit" in msg.lower() or "429" in msg) else 0.5 * (attempt + 1)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                return None
    return None


def check_market_direction(start_date, end_date):
    """(M) Check SPY trend. Returns (status_string, spy_dataframe)."""
    print("Checking market direction (SPY)...")
    spy_df = get_stock_data("SPY", start_date, end_date)

    if spy_df is None or len(spy_df) < 200:
        return "UNKNOWN", None

    price = spy_df["Close"].iloc[-1]
    sma50 = spy_df["Close"].tail(50).mean()
    sma200 = spy_df["Close"].tail(200).mean()

    if price > sma50 > sma200:
        status = "CONFIRMED UPTREND"
    elif price > sma200:
        status = "UPTREND UNDER PRESSURE"
    else:
        status = "DOWNTREND"

    print(f"  SPY ${price:.2f} | 50SMA ${sma50:.2f} | 200SMA ${sma200:.2f} -> {status}")
    return status, spy_df


def compute_relative_strength(stock_df, spy_df):
    """(L) IBD-style relative strength: weighted outperformance vs SPY in pct points.
    Weights recent quarter 2x vs prior quarter 1x."""
    sc = stock_df["Close"]
    sp = spy_df["Close"]

    if len(sc) < 126 or len(sp) < 126:
        return None

    s3m = (sc.iloc[-1] / sc.iloc[-63] - 1) * 100
    s6m = (sc.iloc[-1] / sc.iloc[-126] - 1) * 100
    p3m = (sp.iloc[-1] / sp.iloc[-63] - 1) * 100
    p6m = (sp.iloc[-1] / sp.iloc[-126] - 1) * 100

    stock_w = s3m * 2 + (s6m - s3m)
    spy_w = p3m * 2 + (p6m - p3m)

    return round(stock_w - spy_w, 2)


def analyze_volume(df):
    """(S) Supply/demand via volume patterns.
    Returns (vol_ratio, vol_dryup, vol_surge)."""
    vol = df["Volume"]
    avg10 = vol.tail(10).mean()
    avg50 = vol.tail(50).mean()

    vol_ratio = round(avg10 / avg50, 2) if avg50 > 0 else 0
    vol_dryup = avg10 < avg50 * 0.8
    vol_surge = vol.iloc[-1] > avg50 * 1.5

    return vol_ratio, vol_dryup, vol_surge


def analyze_base(df):
    """(N) Consolidation base quality.
    Returns (base_depth_pct, tightness_pct, near_pivot)."""
    close = df["Close"]
    high_52w = close.tail(255).max()

    recent_low = close.tail(50).min()
    base_depth = round((high_52w - recent_low) / high_52w * 100, 1)

    last20 = close.tail(20)
    tightness = round(last20.std() / last20.mean() * 100, 2)

    consolidation_high = close.tail(50).max()
    pct_from_pivot = (consolidation_high - close.iloc[-1]) / consolidation_high * 100
    near_pivot = pct_from_pivot <= 5

    return base_depth, tightness, near_pivot


def compute_scores(output):
    """Compute composite 0-100 scores using percentile ranks for RS
    and tiered scoring for volume/base quality."""

    # RS score (30 pts): percentile rank within this batch
    output["_rs_score"] = output["RS"].rank(pct=True) * 30

    # Technical score (30 pts): how many of the 6 SMA conditions passed
    output["_tech_score"] = (output["Conds Met"] / 6) * 30

    # Volume score (20 pts): 10 for dry-up, 10 for surge
    output["_vol_score"] = output["Vol Dry-Up"] * 10 + output["Vol Surge"] * 10

    # Base score (20 pts)
    def base_score(row):
        score = 0
        d = row["Base Depth %"]
        if 10 <= d <= 35:
            score += 8
        elif d < 10:
            score += 5
        else:
            score += 2

        t = row["Tightness"]
        if t <= 1.5:
            score += 7
        elif t <= 3.0:
            score += 5
        elif t <= 5.0:
            score += 3
        else:
            score += 1

        if row["Near Pivot"]:
            score += 5
        return score

    output["_base_score"] = output.apply(base_score, axis=1)

    output["Score"] = (
        output["_rs_score"] + output["_tech_score"] + output["_vol_score"] + output["_base_score"]
    ).round().astype(int).clip(0, 100)

    output.drop(columns=["_rs_score", "_tech_score", "_vol_score", "_base_score"], inplace=True)
    return output


def col_to_letter(col_num):
    """1-based column number -> spreadsheet letter (A, B, ... Z, AA, AB, ...)."""
    result = ""
    while col_num > 0:
        col_num, remainder = divmod(col_num - 1, 26)
        result = chr(65 + remainder) + result
    return result


def run_finviz_screen():
    """Run Finviz with expanded filters, fall back to core if too few results."""
    foverview = Overview()
    foverview.set_filter(filters_dict=EXPANDED_FILTERS)
    finviz = foverview.screener_view()

    if isinstance(finviz, pd.DataFrame) and len(finviz) >= 3:
        print(f"Expanded CANSLIM filters returned {len(finviz)} stocks")
        return finviz

    print("Expanded filters too restrictive, falling back to core filters...")
    foverview = Overview()
    foverview.set_filter(filters_dict=CORE_FILTERS)
    finviz = foverview.screener_view()

    if isinstance(finviz, pd.DataFrame):
        print(f"Core filters returned {len(finviz)} stocks")
        return finviz

    return None


def main():
    with open("src/screener/service_account.json") as f:
        sa_json = json.load(f)

    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        sa_json,
        ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"],
    )

    now = dt.datetime.now()
    start = now - dt.timedelta(days=400)

    # ── (M) Market Direction ──
    market_status, spy_df = check_market_direction(start, now)
    if spy_df is None:
        print("Could not fetch SPY data — aborting.")
        return "Could not determine market direction"
    print()

    # ── Finviz Pre-Filter (C, A, I) ──
    finviz = run_finviz_screen()
    if finviz is None:
        return "Could not get stocks from FinViz"

    if "P/E" in finviz.columns:
        finviz.drop(columns=["P/E"], inplace=True)
    if "No." in finviz.columns:
        finviz.drop(columns=["No."], inplace=True)

    stocks = finviz.to_dict("records")
    print(f"Processing {len(stocks)} stocks\n")

    # ── Per-stock analysis loop ──
    stock_data = []
    for i, stock in enumerate(stocks):
        if "Ticker\n\n" in stock:
            stock["Ticker"] = stock.pop("Ticker\n\n")

        ticker = stock["Ticker"]
        print(f"[{i+1}/{len(stocks)}] {ticker}...", end=" ")

        if i > 0:
            time.sleep(0.1)

        try:
            df = get_stock_data(ticker, start, now)
            if df is None:
                print("skipped (no data)")
                continue

            c = df["Close"]
            current = c.iloc[-1]

            sma50 = round(c.tail(50).mean(), 2)
            sma150 = round(c.tail(150).mean(), 2)
            sma200 = round(c.tail(200).mean(), 2)
            sma200_20ago = round(c.iloc[-221:-21].mean(), 2)

            low_52w = round(c.tail(255).min(), 2)
            high_52w = round(c.tail(255).max(), 2)

            cond_1 = current > sma150 > sma200
            cond_2 = sma200 > sma200_20ago
            cond_3 = sma50 > sma150 > sma200
            cond_4 = current > sma50
            cond_5 = current >= 1.3 * low_52w
            cond_6 = current >= 0.75 * high_52w
            conds_met = sum([cond_1, cond_2, cond_3, cond_4, cond_5, cond_6])

            rs = compute_relative_strength(df, spy_df)
            vol_ratio, vol_dryup, vol_surge = analyze_volume(df)
            base_depth, tightness, near_pivot = analyze_base(df)

            stock["RS"] = rs if rs is not None else 0
            stock["Vol Ratio"] = vol_ratio
            stock["Vol Dry-Up"] = int(vol_dryup)
            stock["Vol Surge"] = int(vol_surge)
            stock["Base Depth %"] = base_depth
            stock["Tightness"] = tightness
            stock["Near Pivot"] = int(near_pivot)
            stock["Conds Met"] = conds_met

            stock_data.append(stock)
            print(f"OK  RS:{rs}  VolR:{vol_ratio}  Base:{base_depth}%  Tight:{tightness}")

        except Exception as e:
            print(f"ERROR: {e}")

    if not stock_data:
        print("\nNo stocks processed!")
        return "No stocks processed"

    output = pd.DataFrame(stock_data)
    print(f"\n{len(output)} stocks processed. Computing scores...")

    # ── Composite Score ──
    output = compute_scores(output)
    output.sort_values("Score", ascending=False, inplace=True)
    output.reset_index(drop=True, inplace=True)

    # ── Sharia Check ──
    for idx, row in output.iterrows():
        output.at[idx, "Sharia"] = sharia_status(row["Ticker"])

    # ── Type casting & rounding ──
    int_cols = {"Score": "int", "Conds Met": "int", "Vol Dry-Up": "int",
                "Vol Surge": "int", "Near Pivot": "int", "Volume": "int", "Market Cap": "int"}
    output = output.astype(int_cols)
    output = output.round({"Price": 2, "Change": 4, "RS": 2, "Vol Ratio": 2,
                           "Base Depth %": 1, "Tightness": 2})

    # Record results in the tracker (never blocks the main flow)
    try:
        from tracker import record_screener_run
        tracked = output[output['Sharia'] != 'FAILED']
        tickers_data = [
            {'ticker': row['Ticker'], 'price': row['Price'], 'score': row['Score']}
            for _, row in tracked.iterrows()
        ]
        record_screener_run(tickers_data, 'v2')
    except Exception as e:
        print(f"[Tracker] Warning — could not record run: {e}")

    # ── Column ordering: key metrics first ──
    priority = ["Ticker", "Company", "Score", "RS", "Price", "Change", "Volume",
                "Vol Ratio", "Base Depth %", "Tightness", "Sector", "Industry",
                "Market Cap", "Country", "Sharia"]
    hidden = ["Vol Dry-Up", "Vol Surge", "Near Pivot", "Conds Met"]
    ordered = [c for c in priority if c in output.columns]
    remaining = [c for c in output.columns if c not in ordered and c not in hidden]
    output = output[ordered + remaining + [c for c in hidden if c in output.columns]]

    # ── Write to Google Sheet (CANSLIM tab) ──
    ny_date = dt.datetime.now(pytz.timezone("America/New_York")).strftime("%m-%d-%Y")
    tab_title = f"CANSLIM {ny_date}"

    gc = gspread.authorize(creds)
    gs = gc.open("Stock Screener")

    for ws in gs.worksheets():
        if "CANSLIM" in ws.title:
            gs.del_worksheet(ws)

    sheet = gs.add_worksheet(title=tab_title, rows=1, cols=1)

    set_with_dataframe(
        worksheet=sheet,
        dataframe=output,
        include_index=False,
        include_column_header=True,
        resize=True,
    )

    col_count = len(output.columns)
    first_row = f"A1:{col_to_letter(col_count)}1"

    # Market direction note below data
    status_row = len(output) + 3
    sheet.add_rows(3)
    sheet.update_cell(status_row, 1, f"Market Direction: {market_status}")
    sheet.format(f"A{status_row}", {"textFormat": {"bold": True, "fontSize": 11}})

    # Green header to distinguish from v1's blue
    sheet.format(first_row, {
        "backgroundColor": {"red": 200 / 255, "green": 240 / 255, "blue": 200 / 255},
        "horizontalAlignment": "CENTER",
        "textFormat": {"fontSize": 12, "bold": True},
    })

    # Number formatting
    for col_name, fmt in [
        ("Change", {"numberFormat": {"type": "PERCENT"}}),
        ("Market Cap", {"numberFormat": {"type": "NUMBER", "pattern": '0,,"M"'}}),
        ("Price", {"numberFormat": {"type": "CURRENCY"}}),
        ("Volume", {"numberFormat": {"type": "NUMBER", "pattern": '0.0,,"M"'}}),
    ]:
        cell = sheet.find(col_name)
        if cell:
            letter = col_to_letter(cell.col)
            sheet.format(f"{letter}:{letter}", fmt)

    # Filter (hide FAILED sharia) and sort by Score
    score_cell = sheet.find("Score")
    sharia_cell = sheet.find("Sharia")

    if score_cell and sharia_cell:
        gs.batch_update({"requests": [{"setBasicFilter": {"filter": {
            "range": {"sheetId": sheet.id},
            "filterSpecs": [{
                "filterCriteria": {"hiddenValues": ["FAILED"]},
                "columnIndex": sharia_cell.col - 1,
            }],
            "sortSpecs": [{
                "sortOrder": "DESCENDING",
                "dimensionIndex": score_cell.col - 1,
            }],
        }}}]})

    # Hide detail columns
    for col_name in hidden:
        cell = sheet.find(col_name)
        if cell:
            sheet.hide_columns(cell.col - 1, cell.col)

    sheet.columns_auto_resize(0, col_count)

    print(f"\nDone! Tab '{tab_title}' created. Market: {market_status}")
    return f"CANSLIM v2 complete. Market: {market_status}"


if __name__ == "__main__":
    main()
