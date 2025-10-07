import dash
from dash import html, dcc
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize the Dash app with pages
app = dash.Dash(__name__, use_pages=True)

app.title = "SoupTrader Dashboard"

# Main app layout with navigation
app.layout = html.Div([
    # Navigation header
    html.Div([
        html.H1("SoupTrader", 
                style={'color': '#2c3e50', 'margin': '0', 'display': 'inline-block'}),
        html.Div([
            dcc.Link("My Trading Performance", href="/", 
                    style={'margin': '0 20px', 'color': '#3498db', 'textDecoration': 'none'}),
            dcc.Link("Download Ticker Data", href="/download-ticker-data", 
                    style={'margin': '0 20px', 'color': '#3498db', 'textDecoration': 'none'}),
            dcc.Link("Backtest", href="/backtest", 
                    style={'margin': '0 20px', 'color': '#3498db', 'textDecoration': 'none'})
        ], style={'float': 'right', 'marginTop': '10px'})
    ], style={'padding': '20px', 'borderBottom': '2px solid #ecf0f1', 'backgroundColor': '#f8f9fa'}),
    
    # Page content
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
