from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_df(query):
    conn = sqlite3.connect('/home/ubuntu/souptrader/data/souptrader.db')
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    # Get data from database
    monthly = get_df("SELECT * FROM monthly_summary ORDER BY month")
    quarterly = get_df("SELECT * FROM quarterly_summary ORDER BY quarter")
    yearly = get_df("SELECT * FROM yearly_summary ORDER BY year DESC")
    by_stock = get_df("SELECT * FROM yearly_stock_pnl ORDER BY year DESC, ticker")

    # Create monthly profit line chart
    monthly['month'] = pd.to_datetime(monthly['month'])
    fig_monthly = px.line(monthly, x='month', y='realized_profit',
                         title='Monthly Realized Profits',
                         labels={'realized_profit': 'Profit ($)', 'month': 'Month'})
    fig_monthly.update_layout(
        template='plotly_white',
        hovermode='x unified'
    )

    # Create quarterly performance bar chart
    quarterly['quarter'] = pd.to_datetime(quarterly['quarter'].str.replace('Q', '-'))
    fig_quarterly = px.bar(quarterly, x='quarter', y='realized_profit',
                          title='Quarterly Performance',
                          labels={'realized_profit': 'Profit ($)', 'quarter': 'Quarter'})
    fig_quarterly.update_layout(
        template='plotly_white',
        hovermode='x unified'
    )

    # Create yearly profit pie chart
    fig_yearly = px.pie(yearly, values='realized_profit', names='year',
                       title='Yearly Profit Distribution',
                       labels={'realized_profit': 'Profit ($)', 'year': 'Year'})
    fig_yearly.update_layout(
        template='plotly_white',
        showlegend=True
    )

    # Create stock performance heatmap
    pivot_stock = by_stock.pivot(index='ticker', columns='year', values='profit')
    fig_stock = px.imshow(pivot_stock,
                         title='Stock Performance by Year',
                         labels={'x': 'Year', 'y': 'Ticker', 'color': 'Profit ($)'},
                         color_continuous_scale='RdYlGn')
    fig_stock.update_layout(
        template='plotly_white',
        xaxis_title='Year',
        yaxis_title='Ticker',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Convert figures to JSON for embedding in HTML
    monthly_json = json.dumps(fig_monthly.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
    quarterly_json = json.dumps(fig_quarterly.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
    yearly_json = json.dumps(fig_yearly.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
    stock_json = json.dumps(fig_stock.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "monthly": monthly.to_dict(orient="records"),
        "quarterly": quarterly.to_dict(orient="records"),
        "yearly": yearly.to_dict(orient="records"),
        "by_stock": by_stock.to_dict(orient="records"),
        "monthly_chart": monthly_json,
        "quarterly_chart": quarterly_json,
        "yearly_chart": yearly_json,
        "stock_chart": stock_json
    })
