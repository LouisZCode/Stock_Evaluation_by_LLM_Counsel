
"""
Fake data while we dont have API Calls, just to test for free data injection into LLMS
"""

import random

def _fake_stock_market_data(ticker_symbol: str) -> str:
    ticket_symbol = ticker_symbol.upper()
    #Sadly, API calls are only 25 per day, so will be using mocking data for this exercise:
    lower_price = random.randint(10 , 200)
    higher_price = random.randint(201 , 500)

    pe_ratio = random.randint(10 , 40)

    return f"the ticket symbol {ticket_symbol} has a lowest price of {lower_price}, and highest of {higher_price}, with a pe ratio of {pe_ratio} times per sales"

