import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import os
import sys
import subprocess
import pandas as pd
import exchange_calendars as xcals

# Paths and constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'backtest_data')

# Match downloader date ranges
START_DATE_STR = "2024-09-01"
END_DATE_STR = "2025-08-31"

# Trading calendar
XNYS = xcals.get_calendar('XNYS')


def month_range(start: pd.Timestamp, end: pd.Timestamp):
    result = []
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_month_start = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while current <= end_month_start:
        next_month = (current + pd.offsets.MonthBegin(1))
        month_end = min(end, next_month - pd.Timedelta(nanoseconds=1))
        month_start = max(start, current)
        result.append((month_start, month_end))
        current = next_month
    return result


def daterange_days(start: pd.Timestamp, end: pd.Timestamp):
    days = []
    current = pd.Timestamp(year=start.year, month=start.month, day=start.day, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=end.day, tz="UTC")
    while current <= last:
        days.append(current)
        current = current + pd.Timedelta(days=1)
    return days


def trading_days(start: pd.Timestamp, end: pd.Timestamp):
    # Use exchange calendar sessions (dates are session labels)
    sessions = XNYS.sessions_in_range(start.tz_localize(None), end.tz_localize(None))
    # Convert session labels to UTC midnight timestamps for filename formatting
    return [pd.Timestamp(s).tz_localize('UTC') for s in sessions]


def is_symbol_complete(symbol: str) -> bool:
    sym = symbol.upper()
    start = pd.Timestamp(START_DATE_STR, tz="UTC")
    end = pd.Timestamp(END_DATE_STR + " 23:59:59", tz="UTC")

    # Bars completeness for 1Min and 5Min per-month files
    months = month_range(start, end)
    for timeframe_str in ["1Min", "5Min"]:
        out_dir = os.path.join(DATA_ROOT, sym, timeframe_str)
        expected_files = [os.path.join(out_dir, f"{m_start.strftime('%Y-%m')}.parquet") for m_start, _ in months]
        if not expected_files or not all(os.path.exists(p) for p in expected_files):
            return False

    # NBBO completeness for trading days only (matches downloader behavior)
    days = trading_days(pd.Timestamp(START_DATE_STR, tz="UTC"), pd.Timestamp(END_DATE_STR, tz="UTC"))
    nbbo_dir = os.path.join(DATA_ROOT, sym, "nbbo1s")
    expected_daily = [os.path.join(nbbo_dir, f"{d.strftime('%Y-%m-%d')}.parquet") for d in days]
    if not expected_daily or not all(os.path.exists(p) for p in expected_daily):
        return False

    return True


def list_complete_symbols():
    if not os.path.isdir(DATA_ROOT):
        return []
    symbols = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    complete = [s for s in sorted(symbols) if is_symbol_complete(s)]
    return complete


def render_complete_list(symbols):
    if not symbols:
        return html.P("None yet", style={'textAlign': 'center', 'color': '#7f8c8d'})
    return html.Div([
        html.Span(sym, style={
            'display': 'inline-block',
            'margin': '6px',
            'padding': '6px 10px',
            'borderRadius': '16px',
            'backgroundColor': 'rgba(0,0,0,0.05)',
            'fontSize': '14px'
        }) for sym in symbols
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})


def create_layout():
    complete = list_complete_symbols()
    return html.Div([
        html.H2("Backtesting", style={'textAlign': 'center', 'margin': '0 0 28px'}),

        html.Div([
            html.Div([
                html.Label('Ticker', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(id='ticker-input', type='text', placeholder='Enter ticker (e.g., AAPL)', debounce=True,
                          style={'width': '100%', 'maxWidth': '100%', 'padding': '12px 14px', 'borderRadius': '8px', 'boxSizing': 'border-box'})
            ]),

            html.Div([
                html.Label('API Key', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(id='api-key-input', type='password', placeholder='Enter API key', debounce=True,
                          style={'width': '100%', 'maxWidth': '100%', 'padding': '12px 14px', 'borderRadius': '8px', 'boxSizing': 'border-box'})
            ]),

            html.Div([
                dcc.Checklist(
                    id='force-check',
                    options=[{'label': 'Force re-download', 'value': 'force'}],
                    value=[]
                )
            ]),

            html.Div([
                html.Button('Download', id='download-btn', n_clicks=0,
                            style={'padding': '10px 18px', 'borderRadius': '8px', 'cursor': 'pointer'})
            ], style={'textAlign': 'center'})
        ], style={
            'display': 'grid', 'rowGap': '16px',
            'maxWidth': '480px', 'width': '100%', 'margin': '0 auto 32px', 'padding': '20px', 'borderRadius': '14px',
            'boxShadow': '0 6px 20px rgba(0,0,0,0.08)', 'backgroundColor': 'white', 'boxSizing': 'border-box'
        }),

        html.Div(id='download-status', style={'textAlign': 'center', 'margin': '0 0 32px', 'minHeight': '26px'}),

        html.H3("Tickers we already downloaded", style={'textAlign': 'center', 'margin': '0 0 14px'}),
        html.Div(id='complete-tickers', children=render_complete_list(complete),
                 style={'maxWidth': '760px', 'margin': '0 auto 36px'})
    ], style={'padding': '18px'})


# Create the layout variable that Dash expects
layout = create_layout()


@dash.callback(
    Output('download-status', 'children'),
    Output('complete-tickers', 'children'),
    Input('download-btn', 'n_clicks'),
    State('ticker-input', 'value'),
    State('force-check', 'value'),
    State('api-key-input', 'value'),
    prevent_initial_call=True
)
def handle_download(n_clicks, ticker, force_values, api_key):
    if not ticker or not str(ticker).strip():
        return "Please enter a ticker symbol.", render_complete_list(list_complete_symbols())
    if not api_key or not str(api_key).strip():
        return "Please enter your API key.", render_complete_list(list_complete_symbols())

    sym = str(ticker).strip().upper()
    force = 'force' in (force_values or [])
    api_key_str = str(api_key).strip()

    # Build commands
    py = sys.executable
    bars_cmd = [py, '-m', 'src.backtesting.download_bars', '--symbols', sym]
    nbbo_cmd = [py, '-m', 'src.backtesting.download_nbbo_1s', '--symbol', sym]
    if force:
        bars_cmd.append('--force')
        nbbo_cmd.append('--force')

    try:
        env = os.environ.copy()
        env['MY_API_KEY'] = api_key_str
        # Run both sequentially to keep server resource usage simple
        bars_proc = subprocess.run(bars_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        if bars_proc.returncode != 0:
            msg = f"Bars download failed for {sym}: {bars_proc.stderr.strip() or bars_proc.stdout.strip()}"
            return msg, render_complete_list(list_complete_symbols())

        nbbo_proc = subprocess.run(nbbo_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        if nbbo_proc.returncode != 0:
            msg = f"NBBO download failed for {sym}: {nbbo_proc.stderr.strip() or nbbo_proc.stdout.strip()}"
            return msg, render_complete_list(list_complete_symbols())

        # Success
        status = f"Completed downloads for {sym}."
    except Exception as e:
        status = f"Error: {e}"

    complete = list_complete_symbols()
    return status, render_complete_list(complete)


# Register the page
dash.register_page(__name__, path="/backtest", name="Backtesting")
