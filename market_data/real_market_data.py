
import finnhub
import os
from dotenv import load_dotenv
load_dotenv()
FINHUB_API_KEY = os.getenv("FINHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=FINHUB_API_KEY)

"""
Data pulled form the stock market based on API Calls
"""
def _stock_market_data(ticker : str) -> str:
    ticker_symbol = ticker
    metrics = finnhub_client.company_basic_financials(ticker_symbol, 'all')
    highest_price = metrics["metric"]["52WeekHigh"]
    lowest_price = metrics["metric"]["52WeekLow"]
    pe_ratio = metrics['metric']['peBasicExclExtraTTM']

    print(f"the ticker symbol {ticker_symbol} has a lowest price of {lowest_price}, and highest of {highest_price}, with a pe ratio of {pe_ratio} times per sales")
    return f"the ticker symbol {ticker_symbol} has a lowest price of {lowest_price}, and highest of {highest_price}, with a pe ratio of {pe_ratio} times per sales"