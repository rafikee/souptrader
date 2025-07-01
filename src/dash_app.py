import dash
from dash import html, dcc, dash_table
import pandas as pd
import sqlite3

# Initialize the Dash app
app = dash.Dash(__name__)

app.title = "SoupTrader Dashboard"

def load_data():
    # Connect to SQLite database
    conn = sqlite3.connect('/home/ubuntu/souptrader/data/souptrader.db')
    
    # Load monthly data and sort by month in descending order
    monthly_df = pd.read_sql_query("SELECT * FROM monthly_summary ORDER BY month DESC", conn)
    
    # Create a copy of realized_profit for formatting
    monthly_df['Realized Profit'] = monthly_df['realized_profit'].apply(lambda x: f"{x:,.2f}")
    monthly_df.rename(columns={'month': 'Month'}, inplace=True)
    
    # Load quarterly data
    quarterly_df = pd.read_sql_query("SELECT * FROM quarterly_summary ORDER BY quarter DESC", conn)
    quarterly_df['Realized Profit'] = quarterly_df['realized_profit'].apply(lambda x: f"{x:,.2f}")
    quarterly_df['Trading Capital'] = quarterly_df['cumulative_deposits'].apply(lambda x: f"{x:,.2f}")
    quarterly_df['Annualized Return'] = quarterly_df['annualized_return'].apply(lambda x: f"{x:.2f}%")
    quarterly_df.rename(columns={'quarter': 'Quarter'}, inplace=True)
    
    # Load yearly data
    yearly_df = pd.read_sql_query("SELECT * FROM yearly_summary ORDER BY year DESC", conn)
    yearly_df['Realized Profit'] = yearly_df['realized_profit'].apply(lambda x: f"{x:,.2f}")
    yearly_df['Trading Capital'] = yearly_df['total_deposits'].apply(lambda x: f"{x:,.2f}")
    yearly_df['Annualized Return'] = yearly_df['annualized_return'].apply(lambda x: f"{x:.2f}%")
    yearly_df['Win Rate'] = yearly_df['win_rate'].apply(lambda x: f"{x:.2f}%")
    yearly_df['Avg Trade Duration'] = yearly_df['avg_trade_duration'].apply(lambda x: f"{x:.1f} days")
    yearly_df['Avg Position Size'] = yearly_df['avg_position_size'].apply(lambda x: f"{x:,.2f}")
    yearly_df['Avg Winning Return'] = yearly_df['avg_winning_return_pct'].apply(lambda x: f"{x:.2f}%")
    yearly_df['Avg Losing Return'] = yearly_df['avg_losing_return_pct'].apply(lambda x: f"{x:.2f}%")
    
    conn.close()
    return monthly_df, quarterly_df, yearly_df

# Create the layout
def create_layout():
    # Load fresh data from database on each page load
    monthly_df, quarterly_df, yearly_df = load_data()
    
    # Create columns list, excluding the original realized_profit
    monthly_columns = [
        {"name": col, "id": col} 
        for col in monthly_df.columns 
        if col != 'realized_profit'
    ]
    
    quarterly_columns = [
        {"name": col, "id": col} 
        for col in quarterly_df.columns 
        if col not in ['realized_profit', 'cumulative_deposits', 'annualized_return']
    ]
    
    yearly_columns = [
        {"name": col, "id": col} 
        for col in yearly_df.columns 
        if col not in ['realized_profit', 'total_fees', 'total_dividends', 'options_profit', 
                      'total_deposits', 'annualized_return', 'total_trades_closed', 'win_rate',
                      'avg_trade_duration', 'avg_profit_per_trade', 'avg_position_size',
                      'simple_avg_return', 'position_weighted_return', 'avg_winning_position_size',
                      'avg_winning_return_pct', 'avg_losing_position_size', 'avg_losing_return_pct']
    ]
    
    return html.Div([
        html.H2("SoupTrader Dashboard", style={'textAlign': 'center', 'color': 'green'}),
        html.H3("Monthly Summary Table", style={'textAlign': 'center'}),
        dash_table.DataTable(
            columns=monthly_columns,
            data=monthly_df.to_dict("records"),
            style_table={
                'overflowX': 'auto',
                'maxWidth': '200px',  # Adjusted table width
                'margin': '0 auto',    # Center the table
                'border': '1px solid black',  # Add border to table
                'padding': '5px'     # Add horizontal padding
            },
            style_cell={
                'textAlign': 'center',
                'padding': '5px',      # Reduce cell padding
                'minWidth': '50px',    # Set column width to 50px
                'maxWidth': '50px',    # Set column width to 50px
                'whiteSpace': 'normal', # Allow text wrapping
                'border': '1px solid black'  # Add border to cells
            },
            style_header={
                'backgroundColor': '#f2f2f2',
                'fontWeight': 'bold',
                'border': '1px solid black'  # Add border to header
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{realized_profit} > 0',
                        'column_id': 'Realized Profit'
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'filter_query': '{realized_profit} < 0',
                        'column_id': 'Realized Profit'
                    },
                    'color': 'red'
                }
            ],
            page_size=20  # optional pagination
        ),
        
        html.H3("Quarterly Summary Table", style={'marginTop': '30px', 'textAlign': 'center'}),
        dash_table.DataTable(
            columns=quarterly_columns,
            data=quarterly_df.to_dict("records"),
            style_table={
                'overflowX': 'auto',
                'maxWidth': '400px',  # Adjusted table width
                'margin': '0 auto',    # Center the table
                'border': '1px solid black',  # Add border to table
                'padding': '5px'     # Add horizontal padding
            },
            style_cell={
                'textAlign': 'center',
                'padding': '5px',      # Reduce cell padding
                'minWidth': '50px',    # Set column width to 50px
                'maxWidth': '50px',    # Set column width to 50px
                'whiteSpace': 'normal', # Allow text wrapping
                'border': '1px solid black'  # Add border to cells
            },
            style_header={
                'backgroundColor': '#f2f2f2',
                'fontWeight': 'bold',
                'border': '1px solid black'  # Add border to header
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{realized_profit} > 0',
                        'column_id': 'Realized Profit'
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'filter_query': '{realized_profit} < 0',
                        'column_id': 'Realized Profit'
                    },
                    'color': 'red'
                }
            ],
            page_size=20  # optional pagination
        ),
        
        html.H3("Yearly Summary Table", style={'marginTop': '30px', 'textAlign': 'center'}),
        dash_table.DataTable(
            columns=yearly_columns,
            data=yearly_df.to_dict("records"),
            style_table={
                'overflowX': 'auto',
                'maxWidth': '800px',  # Adjusted table width
                'margin': '0 auto',    # Center the table
                'border': '1px solid black',  # Add border to table
                'padding': '5px'     # Add horizontal padding
            },
            style_cell={
                'textAlign': 'center',
                'padding': '5px',      # Reduce cell padding
                'minWidth': '50px',    # Set column width to 50px
                'maxWidth': '50px',    # Set column width to 50px
                'whiteSpace': 'normal', # Allow text wrapping
                'border': '1px solid black'  # Add border to cells
            },
            style_header={
                'backgroundColor': '#f2f2f2',
                'fontWeight': 'bold',
                'border': '1px solid black'  # Add border to header
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{realized_profit} > 0',
                        'column_id': 'Realized Profit'
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'filter_query': '{realized_profit} < 0',
                        'column_id': 'Realized Profit'
                    },
                    'color': 'red'
                }
            ],
            page_size=20  # optional pagination
        )
    ])



app.layout = create_layout

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 