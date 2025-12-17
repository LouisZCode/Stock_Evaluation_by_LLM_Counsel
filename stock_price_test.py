import finnhub
import os
from dotenv import load_dotenv

load_dotenv()
FINHUB_API_KEY = os.getenv("FINHUB_API_KEY")
ticker = "META"

finnhub_client = finnhub.Client(api_key=FINHUB_API_KEY)

print("This is the current price:")
print(finnhub_client.quote(ticker)["c"])
print()

metrics = finnhub_client.company_basic_financials(ticker, 'all')
print("This is the pe ratio:")
print(metrics['metric']['peBasicExclExtraTTM'])
print()

print("This is the highest price of the year")
print(metrics["metric"]["52WeekHigh"])
print()
print("this is the lowest price of the year")
print(metrics["metric"]["52WeekLow"])
print()
