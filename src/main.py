from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import sqlite3
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_df(query):
    conn = sqlite3.connect('/home/ubuntu/souptrader/data/souptrader.db')
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    monthly = get_df("SELECT * FROM monthly_summary ORDER BY month")
    quarterly = get_df("SELECT * FROM quarterly_summary ORDER BY quarter")
    yearly = get_df("SELECT * FROM yearly_summary ORDER BY year DESC")
    by_stock = get_df("SELECT * FROM yearly_stock_pnl ORDER BY year DESC, ticker")

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "monthly": monthly.to_dict(orient="records"),
        "quarterly": quarterly.to_dict(orient="records"),
        "yearly": yearly.to_dict(orient="records"),
        "by_stock": by_stock.to_dict(orient="records"),
    })
