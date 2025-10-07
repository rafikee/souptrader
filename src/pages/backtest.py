import dash
from dash import html, dcc, Input, Output, State, callback
import json
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_tickers():
    """Load available tickers from downloaded_tickers.json"""
    ticker_file = os.path.join(PROJECT_ROOT, 'src', 'backtesting', 'downloaded_tickers.json')
    try:
        with open(ticker_file, 'r') as f:
            data = json.load(f)
            return data.get('downloaded', [])
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
            
            # Criteria selection
            html.Div([
                html.Label("Select Criteria:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Checklist(
                    id='criteria-checklist',
                    options=[
                        {'label': ' Criteria 1', 'value': 'criteria1'},
                        {'label': ' Criteria 2', 'value': 'criteria2'},
                        {'label': ' Criteria 3', 'value': 'criteria3'},
                        {'label': ' Criteria 4', 'value': 'criteria4'},
                    ],
                    value=[],
                    style={'fontSize': '16px'},
                    labelStyle={'display': 'block', 'marginBottom': '8px'}
                )
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
            'maxWidth': '600px',
            'margin': '0 auto',
            'border': '1px solid #dee2e6'
        }),
        
        # Results section
        html.Div([
            html.H3("Results", style={'marginTop': '40px', 'marginBottom': '20px', 'color': '#34495e', 'textAlign': 'center'}),
            html.Div(
                id='backtest-results',
                children=html.P("Results will appear here after running a backtest.", 
                              style={'textAlign': 'center', 'color': '#7f8c8d', 'fontStyle': 'italic'}),
                style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '30px',
                    'borderRadius': '10px',
                    'maxWidth': '800px',
                    'margin': '0 auto',
                    'border': '1px solid #dee2e6',
                    'minHeight': '200px'
                }
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
    State('criteria-checklist', 'value'),
    prevent_initial_call=True
)
def run_backtest(n_clicks, ticker, criteria):
    if not ticker:
        return html.Div([
            html.P("⚠️ Please select a ticker first.", 
                  style={'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'center'})
        ])
    
    if not criteria:
        return html.Div([
            html.P("⚠️ Please select at least one criteria.", 
                  style={'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'center'})
        ])
    
    # Placeholder result display
    criteria_list = ', '.join([c.replace('criteria', 'Criteria ') for c in criteria])
    
    return html.Div([
        html.H4(f"Backtest Configuration:", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.P([
            html.Strong("Ticker: "), f"{ticker}"
        ], style={'marginBottom': '10px'}),
        html.P([
            html.Strong("Selected Criteria: "), f"{criteria_list}"
        ], style={'marginBottom': '20px'}),
        html.Hr(style={'margin': '20px 0'}),
        html.P("🔄 Backtest logic will be implemented here.", 
              style={'color': '#3498db', 'fontStyle': 'italic', 'textAlign': 'center'})
    ])

# Register the page
dash.register_page(__name__, path="/backtest", name="Backtest")

