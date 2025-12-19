
import finnhub
import os
from dotenv import load_dotenv

#TODO  Decide what to do or how to separate this information. Counsel of LLMs get financials and prices, or would the price only go to the financial advisor LLM?

load_dotenv()
FINHUB_API_KEY = os.getenv("FINHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=FINHUB_API_KEY)


ticker = "TSLA"

print(f"Ticker: {ticker}")
print("This is the current price:")
print(finnhub_client.quote(ticker)["c"])
print()

metrics = finnhub_client.company_basic_financials(ticker, 'all')
print("This is the highest price of the year")
highest_price = metrics["metric"]["52WeekHigh"]
print(highest_price)
print()
print("this is the lowest price of the year")
lowest_price = metrics["metric"]["52WeekLow"]
print(lowest_price)
print()


#basic formulas:
difference = (highest_price - lowest_price)/2
middle_point = highest_price - difference # for visual in the future...
ten_diss = highest_price * .9
twentyfive_diss = highest_price * .75
fifty_diss = highest_price * .5
print("A 10 % disscount is:")
print(ten_diss)
print()
print("A 25 % disscount is:")
print(twentyfive_diss)
print()
print("A 50 % disscount is:")
print(fifty_diss)
print()

print("This is the pe ratio:")
print(metrics['metric']['peBasicExclExtraTTM'])
print()
