import dash
from dash import html

# Create the layout
def create_layout():
    return html.Div([
        html.H2("Backtesting",
                style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
        
        html.Div([
            html.H3("Coming Soon", 
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '48px', 'marginTop': '100px'}),
            html.P("Backtesting interface will be available here soon.",
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '24px', 'marginTop': '20px'})
        ], style={'marginTop': '100px'})
    ])

# Create the layout variable that Dash expects
layout = create_layout()

# Register the page
dash.register_page(__name__, path="/backtest", name="Backtesting")
