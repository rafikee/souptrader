import dash
from dash import html, dcc, Input, Output, State, callback
import json
import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src to path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from backtesting.backtest_runner import BacktestConfig, run_backtest

def load_tickers():
    """Load available tickers from backtest_data folder"""
    backtest_data_path = os.path.join(PROJECT_ROOT, 'data', 'backtest_data')
    try:
        if os.path.exists(backtest_data_path):
            tickers = [d for d in os.listdir(backtest_data_path) 
                      if os.path.isdir(os.path.join(backtest_data_path, d))]
            return sorted(tickers)
        return []
    except Exception as e:
        print(f"Error loading tickers: {e}")
        return []

# Create the layout
def create_layout():
    tickers = load_tickers()
    
    return html.Div([
        html.H2("Backtest Strategy",
                style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
        
        # Configuration section
        html.Div([
            html.H3("Configuration", style={'marginBottom': '20px', 'color': '#34495e'}),
            
            # Ticker selection
            html.Div([
                html.Label("Select Ticker:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='ticker-dropdown',
                    options=[{'label': ticker, 'value': ticker} for ticker in tickers],
                    placeholder="Choose a ticker...",
                    style={'width': '300px'}
                )
            ], style={'marginBottom': '30px'}),
            
            # Strategy selection (Radio buttons)
            html.Div([
                html.Label("Select Strategy:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                dcc.RadioItems(
                    id='strategy-radio',
                    options=[
                        {'label': ' SMA Alignment (Momentum Continuation)', 'value': 'sma_alignment'},
                        {'label': ' Gap Breakout', 'value': 'gap_breakout'},
                    ],
                    value='gap_breakout',
                    style={'fontSize': '16px'},
                    labelStyle={'display': 'block', 'marginBottom': '8px'}
                )
            ], style={'marginBottom': '30px'}),
            
            # Entry filters (Checkboxes)
            html.Div([
                html.Label("Entry Filters (Optional):", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Checklist(
                    id='filters-checklist',
                    options=[
                        {'label': ' SMA Alignment Required (21>50>200)', 'value': 'sma_alignment'},
                        {'label': ' VWAP Filter (price above VWAP)', 'value': 'vwap_filter'},
                        {'label': ' RSI Filter (RSI > 55)', 'value': 'rsi_filter'},
                        {'label': ' Volume Confirmation (1.5x avg)', 'value': 'volume_confirmation'},
                        {'label': ' Body Size Filter (2x avg)', 'value': 'body_size'},
                    ],
                    value=[],
                    style={'fontSize': '14px'},
                    labelStyle={'display': 'block', 'marginBottom': '8px'}
                )
            ], style={'marginBottom': '30px'}),
            
            # Exit parameters
            html.Div([
                html.Label("Exit Parameters:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                
                html.Div([
                    html.Label("Initial Stop Loss (%):", style={'marginRight': '10px'}),
                    dcc.Input(
                        id='stop-loss-input',
                        type='number',
                        value=0.5,
                        step=0.1,
                        min=0,
                        max=10,
                        style={'width': '80px'}
                    ),
                ], style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Label("Trailing Stop (%):", style={'marginRight': '10px'}),
                    dcc.Input(
                        id='trailing-stop-input',
                        type='number',
                        value=0.25,
                        step=0.05,
                        min=0,
                        max=5,
                        style={'width': '80px'}
                    ),
                ], style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Label("Hard Take Profit (%, optional):", style={'marginRight': '10px'}),
                    dcc.Input(
                        id='take-profit-input',
                        type='number',
                        value=None,
                        placeholder='Leave empty to disable',
                        step=0.1,
                        min=0,
                        max=20,
                        style={'width': '150px'}
                    ),
                ], style={'marginBottom': '10px'}),
                
            ], style={'marginBottom': '30px'}),
            
            # Run button
            html.Button(
                'Run Backtest',
                id='run-backtest-btn',
                n_clicks=0,
                style={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'padding': '12px 30px',
                    'fontSize': '16px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold'
                }
            ),
            
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '30px',
            'borderRadius': '10px',
            'maxWidth': '700px',
            'margin': '0 auto',
            'border': '1px solid #dee2e6'
        }),
        
        # Results section with loading spinner
        html.Div([
            html.H3("Results", style={'marginTop': '40px', 'marginBottom': '20px', 'color': '#34495e', 'textAlign': 'center'}),
            dcc.Loading(
                id="loading-backtest",
                type="default",
                children=html.Div(
                    id='backtest-results',
                    children=html.P("Results will appear here after running a backtest.", 
                                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontStyle': 'italic'}),
                    style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '30px',
                        'borderRadius': '10px',
                        'maxWidth': '1200px',
                        'margin': '0 auto',
                        'border': '1px solid #dee2e6',
                        'minHeight': '200px'
                    }
                )
            )
        ])
    ], style={'padding': '20px'})

# Create the layout variable that Dash expects
layout = create_layout

# Callback for running backtest
@callback(
    Output('backtest-results', 'children'),
    Input('run-backtest-btn', 'n_clicks'),
    State('ticker-dropdown', 'value'),
    State('strategy-radio', 'value'),
    State('filters-checklist', 'value'),
    State('stop-loss-input', 'value'),
    State('trailing-stop-input', 'value'),
    State('take-profit-input', 'value'),
    prevent_initial_call=True
)
def run_backtest_callback(n_clicks, ticker, strategy, filters, stop_loss, trailing_stop, take_profit):
    # Validation
    if not ticker:
        return html.P("⚠️ Please select a ticker first.", 
                     style={'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'center'})
    
    if not strategy:
        return html.P("⚠️ Please select a strategy.", 
                     style={'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'center'})
    
    if stop_loss is None or trailing_stop is None:
        return html.P("⚠️ Please set stop loss and trailing stop values.", 
                     style={'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'center'})
    
    # Convert filters list to dict
    filters_dict = {f: True for f in (filters or [])}
    
    # Convert percentages to decimals
    stop_loss_pct = -1 * (stop_loss / 100)  # e.g., 0.5% -> -0.005
    trailing_stop_pct = trailing_stop / 100  # e.g., 0.25% -> 0.0025
    take_profit_pct = (take_profit / 100) if take_profit else None
    
    # Create config
    config = BacktestConfig(
        ticker=ticker,
        strategy=strategy,
        filters=filters_dict,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
        take_profit_pct=take_profit_pct
    )
    
    # Run backtest
    try:
        results = run_backtest(config)
        
        if results.error:
            return html.Div([
                html.H4("❌ Backtest Error", style={'color': '#e74c3c', 'marginBottom': '20px'}),
                html.Pre(
                    '\n'.join(results.log_messages),
                    style={
                        'backgroundColor': '#2c3e50',
                        'color': '#ecf0f1',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'overflow': 'auto',
                        'maxHeight': '600px',
                        'fontSize': '12px',
                        'fontFamily': 'monospace',
                        'whiteSpace': 'pre-wrap'
                    }
                )
            ])
        
        # Display summary card
        summary_card = html.Div([
            html.H4("📊 Backtest Summary", style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
            
            html.Div([
                # Left column
                html.Div([
                    html.P([html.Strong("Total Trades: "), f"{results.total_trades}"]),
                    html.P([html.Strong("Winning Trades: "), f"{results.winning_trades}"]),
                    html.P([html.Strong("Losing Trades: "), f"{results.losing_trades}"]),
                    html.P([html.Strong("Win Rate: "), f"{results.win_rate:.1f}%"]),
                ], style={'flex': '1'}),
                
                # Right column
                html.Div([
                    html.P([html.Strong("Total P&L: "), f"${results.total_pnl:,.2f}"],
                          style={'color': '#27ae60' if results.total_pnl > 0 else '#e74c3c'}),
                    html.P([html.Strong("Total Return: "), f"{results.total_return_pct:+.2f}%"],
                          style={'color': '#27ae60' if results.total_return_pct > 0 else '#e74c3c'}),
                    html.P([html.Strong("Avg Win: "), f"${results.avg_win:,.2f}"]) if results.winning_trades > 0 else None,
                    html.P([html.Strong("Avg Loss: "), f"${results.avg_loss:,.2f}"]) if results.losing_trades > 0 else None,
                ], style={'flex': '1'}),
            ], style={'display': 'flex', 'gap': '40px', 'marginBottom': '20px'}),
            
            # Exit reasons
            html.Div([
                html.Strong("Exit Reasons:"),
                html.Ul([
                    html.Li(f"{reason}: {count}") 
                    for reason, count in results.exit_reasons.items()
                ])
            ]) if results.exit_reasons else None,
            
        ], style={
            'backgroundColor': '#ecf0f1',
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '30px',
            'border': '2px solid #3498db'
        })
        
        # Display full log
        log_output = html.Div([
            html.H4("📝 Detailed Log", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Pre(
                '\n'.join(results.log_messages),
                style={
                    'backgroundColor': '#2c3e50',
                    'color': '#ecf0f1',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'overflow': 'auto',
                    'maxHeight': '600px',
                    'fontSize': '12px',
                    'fontFamily': 'monospace',
                    'whiteSpace': 'pre-wrap'
                }
            )
        ])
        
        return html.Div([summary_card, log_output])
        
    except Exception as e:
        import traceback
        return html.Div([
            html.H4("❌ Unexpected Error", style={'color': '#e74c3c', 'marginBottom': '20px'}),
            html.Pre(
                f"Error: {str(e)}\n\n{traceback.format_exc()}",
                style={
                    'backgroundColor': '#2c3e50',
                    'color': '#ecf0f1',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'overflow': 'auto',
                    'maxHeight': '600px',
                    'fontSize': '12px',
                    'fontFamily': 'monospace',
                    'whiteSpace': 'pre-wrap'
                }
            )
        ])

# Register the page
dash.register_page(__name__, path="/backtest", name="Backtest")
