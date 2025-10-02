import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import os
import sys
import subprocess
import pandas as pd
import exchange_calendars as xcals
from datetime import datetime, timezone
import threading
import json
import time

# Paths and constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'backtest_data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
TICKERS_JSON = os.path.join(PROJECT_ROOT, 'src', 'backtesting', 'downloaded_tickers.json')

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


def load_downloaded_tickers():
    """Load downloaded tickers from JSON file."""
    if not os.path.exists(TICKERS_JSON):
        return [], []
    try:
        with open(TICKERS_JSON, 'r') as f:
            data = json.load(f)
            return data.get('downloaded', []), data.get('incomplete', [])
    except:
        return [], []


def save_downloaded_tickers(complete_tickers, incomplete_tickers=None):
    """Save downloaded tickers to JSON file."""
    os.makedirs(os.path.dirname(TICKERS_JSON), exist_ok=True)
    
    # If incomplete_tickers not provided, load existing incomplete ones
    if incomplete_tickers is None:
        _, incomplete_tickers = load_downloaded_tickers()
    
    data = {
        'downloaded': sorted(complete_tickers),
        'incomplete': sorted(incomplete_tickers),
        'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    with open(TICKERS_JSON, 'w') as f:
        json.dump(data, f, indent=2)


def quick_cleanup_stale_tickers():
    """Quickly sync JSON with filesystem - validate subfolder structure and categorize tickers."""
    if not os.path.isdir(DATA_ROOT):
        # If no data directory exists, clear the JSON file
        save_downloaded_tickers([], [])
        return [], []
    
    # Get all symbols that actually exist in the filesystem (single os.listdir call)
    try:
        filesystem_symbols = set(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
    except OSError:
        # If we can't read the directory, return empty lists
        return [], []
    
    # Load current JSON to check for stale entries
    current_complete, current_incomplete = load_downloaded_tickers()
    
    # Check each ticker for required subfolders and categorize
    valid_complete = []
    valid_incomplete = []
    
    for symbol in filesystem_symbols:
        symbol_path = os.path.join(DATA_ROOT, symbol)
        required_subfolders = ['1Min', '5Min', 'nbbo1s']
        
        # Check if all required subfolders exist
        has_all_subfolders = all(
            os.path.isdir(os.path.join(symbol_path, subfolder)) 
            for subfolder in required_subfolders
        )
        
        if has_all_subfolders:
            valid_complete.append(symbol)
        else:
            valid_incomplete.append(symbol)
    
    # Sort both lists for consistency
    valid_complete.sort()
    valid_incomplete.sort()
    
    # Only update JSON if there were changes
    if (set(valid_complete) != set(current_complete) or 
        set(valid_incomplete) != set(current_incomplete)):
        save_downloaded_tickers(valid_complete, valid_incomplete)
    
    return valid_complete, valid_incomplete


def cleanup_stale_tickers():
    """Remove tickers from JSON that no longer have directories in filesystem."""
    if not os.path.isdir(DATA_ROOT):
        # If no data directory exists, clear the JSON file
        save_downloaded_tickers([], [])
        return [], []
    
    # Get all symbols that actually exist in the filesystem
    filesystem_symbols = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    # Load current JSON to check for stale entries
    current_complete, current_incomplete = load_downloaded_tickers()
    all_json_symbols = set(current_complete + current_incomplete)
    
    # Remove any symbols from JSON that no longer exist in filesystem
    stale_symbols = all_json_symbols - set(filesystem_symbols)
    if stale_symbols:
        print(f"Removing stale tickers from JSON: {sorted(stale_symbols)}")
    
    # Filter out stale symbols from both lists
    valid_complete = [s for s in current_complete if s in filesystem_symbols]
    valid_incomplete = [s for s in current_incomplete if s in filesystem_symbols]
    
    # Update JSON with cleaned lists
    save_downloaded_tickers(valid_complete, valid_incomplete)
    return valid_complete, valid_incomplete


def validate_all_tickers():
    """Validate all tickers in data directory and update JSON."""
    # First do cleanup to remove stale tickers
    valid_complete, valid_incomplete = cleanup_stale_tickers()
    
    if not os.path.isdir(DATA_ROOT):
        return [], []
    
    # Get all symbols that actually exist in the filesystem
    filesystem_symbols = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    complete = []
    incomplete = []
    
    # Only validate symbols that actually exist in the filesystem
    for symbol in sorted(filesystem_symbols):
        if is_symbol_complete(symbol):
            complete.append(symbol)
        else:
            incomplete.append(symbol)
    
    # Update JSON with both complete and incomplete tickers
    save_downloaded_tickers(complete, incomplete)
    return complete, incomplete


def add_downloaded_ticker(symbol):
    """Add a newly downloaded ticker to the JSON if it's complete."""
    current_complete, current_incomplete = load_downloaded_tickers()
    
    if is_symbol_complete(symbol):
        # Move from incomplete to complete if it was there
        if symbol in current_incomplete:
            current_incomplete.remove(symbol)
        if symbol not in current_complete:
            current_complete.append(symbol)
        save_downloaded_tickers(current_complete, current_incomplete)
        return True
    else:
        # Add to incomplete if not already there
        if symbol not in current_incomplete and symbol not in current_complete:
            current_incomplete.append(symbol)
            save_downloaded_tickers(current_complete, current_incomplete)
        return False


def render_ticker_list(symbols, empty_text="None yet"):
    if not symbols:
        return html.P(empty_text, style={'textAlign': 'center', 'color': '#7f8c8d'})
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




def update_download_status(status, symbol=None, error=None):
    """Update the download status file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    status_file = os.path.join(LOGS_DIR, 'download_status.json')
    
    status_data = {
        "status": status,
        "symbol": symbol,
        "start_time": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        "error": error
    }
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)


def get_download_status():
    """Get current download status from file."""
    status_file = os.path.join(LOGS_DIR, 'download_status.json')
    if not os.path.exists(status_file):
        return {"status": "idle", "symbol": None, "start_time": None, "error": None}
    
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except:
        return {"status": "idle", "symbol": None, "start_time": None, "error": None}


def start_new_log_session():
    """Start a new log session by clearing both log files."""
    # Clear both log files to start fresh session
    for log_name in ['bars_download.log', 'nbbo_download.log']:
        log_file = os.path.join(LOGS_DIR, log_name)
        if os.path.exists(log_file):
            os.remove(log_file)


def download_worker(symbol, force, api_key):
    """Background worker to run downloads."""
    try:
        print(f"Starting download worker for {symbol}")
        # Update status to downloading
        update_download_status("downloading", symbol)
        
        # Start new log session for this download
        start_new_log_session()
        
        # Build commands
        py = sys.executable
        bars_cmd = [py, '-m', 'src.backtesting.download_bars', '--symbols', symbol]
        nbbo_cmd = [py, '-m', 'src.backtesting.download_nbbo_1s', '--symbol', symbol]
        if force:
            bars_cmd.append('--force')
            nbbo_cmd.append('--force')

        env = os.environ.copy()
        env['MY_API_KEY'] = api_key
        
        print(f"Running bars download for {symbol}")
        # Run bars download
        bars_proc = subprocess.run(bars_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        
        if bars_proc.returncode != 0:
            error_msg = f"Bars download failed: {bars_proc.stderr.strip() or bars_proc.stdout.strip()}"
            print(f"Bars download error: {error_msg}")
            update_download_status("failed", symbol, error_msg)
            return

        print(f"Running NBBO download for {symbol}")
        # Run NBBO download
        nbbo_proc = subprocess.run(nbbo_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        
        if nbbo_proc.returncode != 0:
            error_msg = f"NBBO download failed: {nbbo_proc.stderr.strip() or nbbo_proc.stdout.strip()}"
            print(f"NBBO download error: {error_msg}")
            update_download_status("failed", symbol, error_msg)
            return

        print(f"Download completed successfully for {symbol}")
        # Success - add to downloaded tickers (we just downloaded, so it's complete)
        current_complete, current_incomplete = load_downloaded_tickers()
        if symbol not in current_complete:
            current_complete.append(symbol)
        if symbol in current_incomplete:
            current_incomplete.remove(symbol)
        save_downloaded_tickers(current_complete, current_incomplete)
        update_download_status("complete", symbol)
        print(f"Status updated to complete for {symbol}")
        
    except Exception as e:
        error_msg = f"Exception during download: {str(e)}"
        print(f"Download worker exception: {error_msg}")
        update_download_status("failed", symbol, error_msg)
    finally:
        # Always reset status to idle after completion (success or failure)
        # This ensures we don't get stuck in "downloading" state
        print(f"Resetting download status to idle")
        update_download_status("idle")


def create_layout():
    # Load initial data and do quick cleanup to avoid showing stale tickers
    downloaded, incomplete = quick_cleanup_stale_tickers()
    status_data = get_download_status()
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

        html.Div(id='download-status', 
                 children=get_status_message(status_data),
                 style={'textAlign': 'center', 'margin': '0 0 32px', 'minHeight': '26px'}),

        html.H3("Tickers we already downloaded", style={'textAlign': 'center', 'margin': '0 0 14px'}),
        html.Div(id='downloaded-tickers', children=render_ticker_list(downloaded),
                 style={'maxWidth': '760px', 'margin': '0 auto 20px'}),

        html.Div(id='missing-tickers-section', 
                 children=get_missing_tickers_section(incomplete),
                 style={'maxWidth': '760px', 'margin': '0 auto 20px'}),
        
        html.Div([
            html.Button('Validate Downloads', id='validate-btn', n_clicks=0,
                        style={'padding': '8px 16px', 'borderRadius': '8px', 'cursor': 'pointer', 'marginRight': '10px'}),
            html.Button('Reset Download Status', id='reset-status-btn', n_clicks=0,
                        style={'padding': '8px 16px', 'borderRadius': '8px', 'cursor': 'pointer', 'backgroundColor': '#ff6b6b', 'color': 'white', 'border': 'none'})
        ], style={'textAlign': 'center', 'margin': '20px 0'}),

        html.Div(id='validation-status', style={'textAlign': 'center', 'margin': '10px 0', 'minHeight': '20px'}),
        
        # Hidden div to trigger periodic updates
        dcc.Interval(
            id='interval-component',
            interval=3000,  # Update every 3 seconds
            n_intervals=0
        )
    ], style={'padding': '18px'})


def get_status_message(status_data):
    """Get status message based on current download status."""
    if status_data['status'] == 'downloading':
        return f"Downloading {status_data['symbol']}... (started at {status_data['start_time']})"
    elif status_data['status'] == 'complete':
        return f"Download completed for {status_data['symbol']}!"
    elif status_data['status'] == 'failed':
        return f"Download failed for {status_data['symbol']}: {status_data['error']}"
    else:
        return ""


def get_missing_tickers_section(incomplete_tickers):
    """Get the missing tickers section, only show if there are any."""
    if not incomplete_tickers:
        return html.Div()
    
    return html.Div([
        html.H3("Tickers missing data", style={'textAlign': 'center', 'margin': '0 0 14px', 'color': '#e74c3c'}),
        render_ticker_list(incomplete_tickers, "None")
    ])


# Create the layout variable that Dash expects
layout = create_layout


@dash.callback(
    Output('download-status', 'children'),
    Output('downloaded-tickers', 'children'),
    Output('missing-tickers-section', 'children'),
    Output('ticker-input', 'value'),
    Output('api-key-input', 'value'),
    Input('download-btn', 'n_clicks'),
    State('ticker-input', 'value'),
    State('force-check', 'value'),
    State('api-key-input', 'value'),
    prevent_initial_call=True
)
def handle_download(n_clicks, ticker, force_values, api_key):
    if not ticker or not str(ticker).strip():
        downloaded, incomplete = load_downloaded_tickers()
        return "Please enter a ticker symbol.", render_ticker_list(downloaded), get_missing_tickers_section(incomplete), None, None
    if not api_key or not str(api_key).strip():
        downloaded, incomplete = load_downloaded_tickers()
        return "Please enter your API key.", render_ticker_list(downloaded), get_missing_tickers_section(incomplete), None, None

    sym = str(ticker).strip().upper()
    force = 'force' in (force_values or [])
    api_key_str = str(api_key).strip()

    # Check if already downloading
    current_status = get_download_status()
    if current_status['status'] == 'downloading':
        downloaded, incomplete = load_downloaded_tickers()
        return f"Download already in progress for {current_status['symbol']}. Please wait.", render_ticker_list(downloaded), get_missing_tickers_section(incomplete), None, None

    # Start background download
    thread = threading.Thread(target=download_worker, args=(sym, force, api_key_str))
    thread.daemon = True
    thread.start()

    # Return immediately and clear input fields
    status = f"Download started for {sym}. Status will update automatically."
    downloaded, incomplete = load_downloaded_tickers()
    return status, render_ticker_list(downloaded), get_missing_tickers_section(incomplete), "", ""


@dash.callback(
    Output('download-status', 'children', allow_duplicate=True),
    Output('downloaded-tickers', 'children', allow_duplicate=True),
    Output('missing-tickers-section', 'children', allow_duplicate=True),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def update_status(n_intervals):
    """Update status based on download progress."""
    status_data = get_download_status()
    downloaded, incomplete = load_downloaded_tickers()
    
    if status_data['status'] == 'idle':
        return "", render_ticker_list(downloaded), get_missing_tickers_section(incomplete)
    elif status_data['status'] == 'downloading':
        return f"Downloading {status_data['symbol']}... (started at {status_data['start_time']})", render_ticker_list(downloaded), get_missing_tickers_section(incomplete)
    elif status_data['status'] == 'complete':
        # Clear status after showing completion and refresh ticker list
        update_download_status("idle")
        downloaded, incomplete = load_downloaded_tickers()  # Refresh in case new ticker was added
        return f"Download completed for {status_data['symbol']}!", render_ticker_list(downloaded), get_missing_tickers_section(incomplete)
    elif status_data['status'] == 'failed':
        # Clear status after showing error
        update_download_status("idle")
        return f"Download failed for {status_data['symbol']}: {status_data['error']}", render_ticker_list(downloaded), get_missing_tickers_section(incomplete)
    
    return "", render_ticker_list(downloaded), get_missing_tickers_section(incomplete)


@dash.callback(
    Output('downloaded-tickers', 'children', allow_duplicate=True),
    Output('missing-tickers-section', 'children', allow_duplicate=True),
    Output('validation-status', 'children'),
    Input('validate-btn', 'n_clicks'),
    prevent_initial_call=True
)
def validate_downloads(n_clicks):
    """Validate all downloads and update ticker lists."""
    if n_clicks == 0:
        return html.Div(), html.Div(), ""
    
    complete, incomplete = validate_all_tickers()
    
    # Show validation completion message with timestamp
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    status_msg = f"Validation completed at {timestamp} - Found {len(complete)} complete and {len(incomplete)} incomplete tickers"
    
    return render_ticker_list(complete), get_missing_tickers_section(incomplete), status_msg


@dash.callback(
    Output('download-status', 'children', allow_duplicate=True),
    Output('validation-status', 'children', allow_duplicate=True),
    Input('reset-status-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_download_status(n_clicks):
    """Reset download status to idle."""
    if n_clicks == 0:
        return "", ""
    
    # Reset status to idle
    update_download_status("idle")
    
    # Show reset confirmation message
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    status_msg = f"Download status reset to idle at {timestamp}"
    
    return "Ready to download", status_msg


# Register the page
dash.register_page(__name__, path="/backtest", name="Backtesting")
