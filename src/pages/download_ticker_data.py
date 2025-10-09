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

# Import shared validation utilities
import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backtesting.data_validation import is_symbol_complete, get_validation_summary

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


# Validation functions now imported from src.backtesting.data_validation


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
        # Use the shared validation function
        if is_symbol_complete(symbol):
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

        # Load environment variables from .env file
        from dotenv import load_dotenv
        env_file = os.path.join(PROJECT_ROOT, '.env')
        load_dotenv(env_file)
        
        env = os.environ.copy()
        # Get both API credentials from environment (.env file)
        alpaca_key_id = os.getenv('APCA_API_KEY_ID')
        alpaca_secret = os.getenv('APCA_API_SECRET_KEY')
        
        print(f"DEBUG: alpaca_key_id found: {alpaca_key_id is not None}")
        print(f"DEBUG: alpaca_secret found: {alpaca_secret is not None}")
        
        if alpaca_key_id:
            env['APCA_API_KEY_ID'] = alpaca_key_id
        elif api_key:
            # Fallback to UI input if env var not found
            env['APCA_API_KEY_ID'] = api_key
        
        if alpaca_secret:
            env['APCA_API_SECRET_KEY'] = alpaca_secret
        else:
            print("WARNING: APCA_API_SECRET_KEY not found in environment")
        
        print(f"Running bars download for {symbol} (5Min + Daily)")
        # Run bars download (both 5Min and Daily)
        bars_proc = subprocess.run(bars_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        
        # Log the captured output for debugging
        print(f"Bars download stdout: {bars_proc.stdout}")
        print(f"Bars download stderr: {bars_proc.stderr}")
        print(f"Bars download return code: {bars_proc.returncode}")
        
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
        html.H2("Download Ticker Data", style={'textAlign': 'center', 'margin': '0 0 28px'}),

        html.Div([
            html.Div([
                html.Label('Ticker(s)', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(id='ticker-input', type='text', placeholder='Enter ticker(s) (e.g., AAPL or SMR, MVST, ASTS)', debounce=True,
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

        html.Div(id='downloaded-tickers', children=get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'),
                 style={'maxWidth': '800px', 'margin': '0 auto 40px'}),

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


def get_ticker_details_section(tickers, title, title_color='#2c3e50'):
    """Get a section with detailed validation info for tickers."""
    if not tickers:
        return html.Div()
    
    # Get detailed validation info for each ticker
    ticker_details = []
    for ticker in tickers:
        summary = get_validation_summary(ticker)
        
        # Create detail rows for this ticker
        detail_html = html.Div([
            html.Div([
                html.Strong(ticker, style={'fontSize': '16px', 'color': '#2c3e50'}),
                html.Div([
                    html.Div([
                        html.Span("1Min: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{summary['1Min']['found']} months", 
                                 style={'color': '#e74c3c' if summary['1Min']['found'] == 0 else '#2c3e50'})
                    ], style={'display': 'inline-block', 'marginRight': '15px'}),
                    html.Div([
                        html.Span("5Min: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{summary['5Min']['found']} months", 
                                 style={'color': '#e74c3c' if summary['5Min']['found'] == 0 else '#2c3e50'})
                    ], style={'display': 'inline-block', 'marginRight': '15px'}),
                    html.Div([
                        html.Span("Daily: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{summary['Daily']['found']} months", 
                                 style={'color': '#e74c3c' if summary['Daily']['found'] == 0 else '#2c3e50'})
                    ], style={'display': 'inline-block', 'marginRight': '15px'}),
                    html.Div([
                        html.Span("NBBO: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{summary['NBBO']['found']} days", 
                                 style={'color': '#e74c3c' if summary['NBBO']['found'] == 0 else '#2c3e50'})
                    ], style={'display': 'inline-block'})
                ], style={'fontSize': '14px', 'marginTop': '5px', 'color': '#555'})
            ], style={
                'padding': '12px',
                'margin': '8px',
                'backgroundColor': '#fff',
                'borderRadius': '8px',
                'border': '1px solid #e0e0e0',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
            })
        ])
        ticker_details.append(detail_html)
    
    return html.Div([
        html.H3(title, style={'textAlign': 'center', 'margin': '0 0 14px', 'color': title_color}),
        html.Div(ticker_details, style={'maxWidth': '800px', 'margin': '0 auto'})
    ])


def get_missing_tickers_section(incomplete_tickers):
    """Get the missing tickers section with detailed validation info."""
    return get_ticker_details_section(incomplete_tickers, "Tickers with incomplete data", '#e74c3c')


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
        return "Please enter a ticker symbol.", get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete), None, None
    # API key is now optional - will use .env file if not provided
    # if not api_key or not str(api_key).strip():
    #     downloaded, incomplete = load_downloaded_tickers()
    #     return "Please enter your API key.", get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete), None, None

    # Split tickers by comma to support multiple symbols
    symbols = [s.strip().upper() for s in str(ticker).split(",") if s.strip()]
    
    if not symbols:
        downloaded, incomplete = load_downloaded_tickers()
        return "Please enter valid ticker symbol(s).", get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete), None, None
    
    force = 'force' in (force_values or [])
    api_key_str = str(api_key).strip() if api_key else ""

    # Check if already downloading
    current_status = get_download_status()
    if current_status['status'] == 'downloading':
        downloaded, incomplete = load_downloaded_tickers()
        return f"Download already in progress for {current_status['symbol']}. Please wait.", render_ticker_list(downloaded), get_missing_tickers_section(incomplete), None, None

    # Start background download for each symbol
    # We use a simple sequential approach with delays to avoid race conditions
    def download_all_symbols():
        import time
        for sym in symbols:
            download_worker(sym, force, api_key_str)
            # Small delay between symbols to ensure proper status tracking
            if sym != symbols[-1]:  # Don't delay after the last one
                time.sleep(1)
    
    thread = threading.Thread(target=download_all_symbols)
    thread.daemon = True
    thread.start()

    # Return immediately and clear input fields
    if len(symbols) == 1:
        status = f"Download started for {symbols[0]}. Status will update automatically."
    else:
        status = f"Download started for {len(symbols)} symbols: {', '.join(symbols)}. Status will update automatically."
    downloaded, incomplete = load_downloaded_tickers()
    return status, get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete), "", ""


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
        return "", get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete)
    elif status_data['status'] == 'downloading':
        return f"Downloading {status_data['symbol']}... (started at {status_data['start_time']})", get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete)
    elif status_data['status'] == 'complete':
        # Clear status after showing completion and refresh ticker list
        update_download_status("idle")
        downloaded, incomplete = load_downloaded_tickers()  # Refresh in case new ticker was added
        return f"Download completed for {status_data['symbol']}!", get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete)
    elif status_data['status'] == 'failed':
        # Clear status after showing error
        update_download_status("idle")
        return f"Download failed for {status_data['symbol']}: {status_data['error']}", get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete)
    
    return "", get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete)


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
        # Don't clear the UI on initial load - just return current state
        downloaded, incomplete = load_downloaded_tickers()
        return get_ticker_details_section(downloaded, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete), ""
    
    complete, incomplete = validate_all_tickers()
    
    # Show validation completion message with timestamp
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    status_msg = f"Validation completed at {timestamp} - Found {len(complete)} complete and {len(incomplete)} incomplete tickers"
    
    return get_ticker_details_section(complete, "Tickers already downloaded", '#27ae60'), get_missing_tickers_section(incomplete), status_msg


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
dash.register_page(__name__, path="/download-ticker-data", name="Download Ticker Data")

