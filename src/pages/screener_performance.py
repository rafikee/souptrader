import dash
from dash import html, dcc, dash_table, callback, Input, Output
import pandas as pd
import sqlite3
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'souptrader.db')

HYPOTHETICAL_INVESTMENT = 10_000

dash.register_page(__name__, path="/screener-performance", name="Screener Performance")


def _conn():
    return sqlite3.connect(DB_PATH)


def _get_open_positions(version):
    conn = _conn()
    signals = pd.read_sql_query(
        "SELECT ticker, entry_date, entry_price "
        "FROM screener_signals "
        "WHERE screener_version = ? AND exit_date IS NULL "
        "ORDER BY entry_date",
        conn, params=[version],
    )
    if signals.empty:
        conn.close()
        return pd.DataFrame()

    latest = pd.read_sql_query(
        "SELECT t.ticker, t.close_price AS current_price, t.score, t.date AS last_seen "
        "FROM screener_tracker t "
        "INNER JOIN ("
        "  SELECT ticker, MAX(date) AS max_date "
        "  FROM screener_tracker WHERE screener_version = ? GROUP BY ticker"
        ") m ON t.ticker = m.ticker AND t.date = m.max_date "
        "WHERE t.screener_version = ?",
        conn, params=[version, version],
    )
    conn.close()

    df = signals.merge(latest, on='ticker', how='left')
    today = datetime.now()
    df['days_held'] = df['entry_date'].apply(
        lambda d: (today - datetime.strptime(d, '%Y-%m-%d')).days
    )
    df['shares'] = HYPOTHETICAL_INVESTMENT / df['entry_price']
    df['current_value'] = df['shares'] * df['current_price']
    df['pnl'] = df['current_value'] - HYPOTHETICAL_INVESTMENT
    df['pnl_pct'] = (df['current_price'] - df['entry_price']) / df['entry_price'] * 100
    return df


def _get_closed_positions(version):
    conn = _conn()
    df = pd.read_sql_query(
        "SELECT ticker, entry_date, exit_date, entry_price, exit_price, return_pct "
        "FROM screener_signals "
        "WHERE screener_version = ? AND exit_date IS NOT NULL "
        "ORDER BY exit_date DESC",
        conn, params=[version],
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df['days_held'] = df.apply(
        lambda r: (datetime.strptime(r['exit_date'], '%Y-%m-%d')
                   - datetime.strptime(r['entry_date'], '%Y-%m-%d')).days,
        axis=1,
    )
    df['shares'] = HYPOTHETICAL_INVESTMENT / df['entry_price']
    df['pnl'] = df['shares'] * (df['exit_price'] - df['entry_price'])
    return df


# ── Styling constants ──

CARD_STYLE = {
    'display': 'inline-block', 'width': '180px', 'padding': '18px 14px',
    'margin': '8px', 'borderRadius': '8px', 'textAlign': 'center',
    'backgroundColor': '#fff', 'boxShadow': '0 1px 4px rgba(0,0,0,.1)',
}
CARD_LABEL = {'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '4px'}
CARD_VALUE = {'fontSize': '22px', 'fontWeight': 'bold', 'color': '#2c3e50'}

TABLE_STYLE = {
    'overflowX': 'auto', 'margin': '0 auto',
    'border': '1px solid #ddd',
}
CELL_STYLE = {
    'textAlign': 'center', 'padding': '8px', 'fontSize': '13px',
    'border': '1px solid #eee', 'whiteSpace': 'normal',
}
HEADER_STYLE = {
    'backgroundColor': '#f2f2f2', 'fontWeight': 'bold',
    'border': '1px solid #ddd', 'fontSize': '13px',
}


def _summary_cards(open_df, closed_df):
    n_open = len(open_df)
    total_deployed = n_open * HYPOTHETICAL_INVESTMENT
    current_value = open_df['current_value'].sum() if n_open else 0
    unrealized_pnl = open_df['pnl'].sum() if n_open else 0
    unrealized_pct = (unrealized_pnl / total_deployed * 100) if total_deployed else 0

    n_closed = len(closed_df)
    wins = len(closed_df[closed_df['return_pct'] > 0]) if n_closed else 0
    win_rate = (wins / n_closed * 100) if n_closed else 0
    realized_pnl = closed_df['pnl'].sum() if n_closed else 0
    avg_return = closed_df['return_pct'].mean() if n_closed else 0

    def card(label, value, color=None):
        style = {**CARD_VALUE}
        if color:
            style['color'] = color
        return html.Div([
            html.Div(label, style=CARD_LABEL),
            html.Div(value, style=style),
        ], style=CARD_STYLE)

    pnl_color = '#27ae60' if unrealized_pnl >= 0 else '#e74c3c'
    realized_color = '#27ae60' if realized_pnl >= 0 else '#e74c3c'

    return html.Div([
        card("Open Positions", str(n_open)),
        card("Capital Deployed", f"${total_deployed:,.0f}"),
        card("Current Value", f"${current_value:,.0f}"),
        card("Unrealized P&L", f"${unrealized_pnl:,.0f} ({unrealized_pct:+.1f}%)", pnl_color),
        card("Closed Trades", str(n_closed)),
        card("Win Rate", f"{win_rate:.0f}%" if n_closed else "—"),
        card("Realized P&L", f"${realized_pnl:,.0f}" if n_closed else "—", realized_color if n_closed else None),
        card("Avg Return", f"{avg_return:+.1f}%" if n_closed else "—"),
    ], style={'textAlign': 'center', 'marginBottom': '30px'})


def _format_open_table(df):
    if df.empty:
        return html.P("No open positions.", style={'textAlign': 'center', 'color': '#95a5a6'})

    display = pd.DataFrame({
        'Ticker': df['ticker'],
        'Entry Date': df['entry_date'],
        'Days Held': df['days_held'],
        'Entry Price': df['entry_price'].apply(lambda x: f"${x:,.2f}"),
        'Current Price': df['current_price'].apply(lambda x: f"${x:,.2f}"),
        'Shares ($10K)': df['shares'].apply(lambda x: f"{x:,.1f}"),
        'Current Value': df['current_value'].apply(lambda x: f"${x:,.0f}"),
        'P&L ($)': df['pnl'].apply(lambda x: f"${x:+,.0f}"),
        'P&L (%)': df['pnl_pct'].apply(lambda x: f"{x:+.1f}%"),
        '_pnl_raw': df['pnl'],
    })
    if 'score' in df.columns and df['score'].notna().any():
        display.insert(1, 'Score', df['score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—"))

    columns = [{"name": c, "id": c} for c in display.columns if not c.startswith('_')]

    return dash_table.DataTable(
        columns=columns,
        data=display.to_dict("records"),
        style_table=TABLE_STYLE,
        style_cell=CELL_STYLE,
        style_header=HEADER_STYLE,
        style_data_conditional=[
            {'if': {'filter_query': '{_pnl_raw} > 0', 'column_id': 'P&L ($)'}, 'color': '#27ae60', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{_pnl_raw} < 0', 'column_id': 'P&L ($)'}, 'color': '#e74c3c', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{_pnl_raw} > 0', 'column_id': 'P&L (%)'}, 'color': '#27ae60', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{_pnl_raw} < 0', 'column_id': 'P&L (%)'}, 'color': '#e74c3c', 'fontWeight': 'bold'},
        ],
        sort_action='native',
        page_size=50,
    )


def _format_closed_table(df):
    if df.empty:
        return html.P("No closed trades yet.", style={'textAlign': 'center', 'color': '#95a5a6'})

    display = pd.DataFrame({
        'Ticker': df['ticker'],
        'Entry Date': df['entry_date'],
        'Exit Date': df['exit_date'],
        'Days Held': df['days_held'],
        'Entry Price': df['entry_price'].apply(lambda x: f"${x:,.2f}"),
        'Exit Price': df['exit_price'].apply(lambda x: f"${x:,.2f}"),
        'Return (%)': df['return_pct'].apply(lambda x: f"{x:+.1f}%"),
        'P&L ($)': df['pnl'].apply(lambda x: f"${x:+,.0f}"),
        '_return_raw': df['return_pct'],
    })

    columns = [{"name": c, "id": c} for c in display.columns if not c.startswith('_')]

    return dash_table.DataTable(
        columns=columns,
        data=display.to_dict("records"),
        style_table=TABLE_STYLE,
        style_cell=CELL_STYLE,
        style_header=HEADER_STYLE,
        style_data_conditional=[
            {'if': {'filter_query': '{_return_raw} > 0', 'column_id': 'Return (%)'}, 'color': '#27ae60', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{_return_raw} < 0', 'column_id': 'Return (%)'}, 'color': '#e74c3c', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{_return_raw} > 0', 'column_id': 'P&L ($)'}, 'color': '#27ae60', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{_return_raw} < 0', 'column_id': 'P&L ($)'}, 'color': '#e74c3c', 'fontWeight': 'bold'},
        ],
        sort_action='native',
        page_size=50,
    )


def create_layout():
    return html.Div([
        html.H2("Screener Performance",
                 style={'textAlign': 'center', 'marginBottom': '10px', 'color': '#2c3e50'}),
        html.P("Track hypothetical $10K positions based on screener appearances.",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '25px'}),

        html.Div([
            html.Label("Screener Version:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='sp-version-dropdown',
                options=[
                    {'label': 'v1 — Trend Template', 'value': 'v1'},
                    {'label': 'v2 — CANSLIM', 'value': 'v2'},
                ],
                value='v2',
                clearable=False,
                style={'width': '250px', 'display': 'inline-block', 'verticalAlign': 'middle'},
            ),
        ], style={'textAlign': 'center', 'marginBottom': '25px'}),

        html.Div(id='sp-summary-cards'),

        html.H3("Open Positions", style={'textAlign': 'center', 'marginTop': '10px', 'color': '#2c3e50'}),
        html.Div(id='sp-open-table'),

        html.Hr(style={'margin': '40px 0'}),

        html.Div([
            html.H3("Closed Trades", style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.Div([
                html.Label("Filter by ticker:", style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='sp-ticker-filter',
                    placeholder="All tickers",
                    style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                ),
            ], style={'textAlign': 'center', 'marginBottom': '15px'}),
        ]),
        html.Div(id='sp-closed-table'),

    ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})


layout = create_layout


@callback(
    Output('sp-summary-cards', 'children'),
    Output('sp-open-table', 'children'),
    Output('sp-closed-table', 'children'),
    Output('sp-ticker-filter', 'options'),
    Input('sp-version-dropdown', 'value'),
    Input('sp-ticker-filter', 'value'),
)
def update_page(version, ticker_filter):
    open_df = _get_open_positions(version)
    closed_df = _get_closed_positions(version)

    ticker_options = [{'label': t, 'value': t}
                      for t in sorted(closed_df['ticker'].unique())] if not closed_df.empty else []

    filtered_closed = closed_df
    if ticker_filter and not closed_df.empty:
        filtered_closed = closed_df[closed_df['ticker'] == ticker_filter]

    return (
        _summary_cards(open_df, filtered_closed),
        _format_open_table(open_df),
        _format_closed_table(filtered_closed),
        ticker_options,
    )
