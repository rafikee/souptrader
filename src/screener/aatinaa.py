# Description: Aatinaa API
# Check if a stock is sharia compliant or not.

from requests import post

URL = 'https://api.aatinaa.co/graphql'

def sharia_status(ticker):

    body = {"operationName":"getStockDetail","variables":{"ticker":ticker},"query":"query getStockDetail($ticker: String!) {\n  getStockDetail(ticker: $ticker) {ticker\n    securityName\n    sector\n    totalDebt\n    totalCash\n    totalDeposit\n    totalReceivables\n    totalAssets\n    totalInterestIncome\n    totalRevenue\n    interestBearingDebt\n    interestBearingSecurities\n    interestVsRevenue\n    interestBearingDebtStatus\n    interestBearingSecuritiesStatus\n    interestVsRevenueStatus\n    redListStatus\n    shariaCompliantStatus\n    marketCap\n    createdAt\n    modifiedAt\n  }\n}\n"}
    
    try:
        data = post(URL, json=body).json()['data']['getStockDetail']

        debt = data.get('interestBearingDebtStatus', None)
        securities = data.get('interestBearingSecuritiesStatus', None)
        revenue = data.get('interestVsRevenueStatus', None)
        sharia_compliant = data.get('shariaCompliantStatus', None)

        if sharia_compliant == 'PASSED':
            return "COMPLIANT"
        elif debt == 'PASSED' and securities == 'PASSED' and revenue == 'PASSED':
            return "QUESTIONABLE"
        else:
            return "FAILED"
    except Exception as e:
        return "UNKNOWN"