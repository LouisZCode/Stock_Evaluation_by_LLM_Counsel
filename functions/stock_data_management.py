
"""
Here all the tools and functions related to the manipulation of the stock data and stock database

You can find:

_stock_market_data, stock_market_data_tool, _save_stock_evals, 
ticker_admin_tool, review_stock_data
"""

import random
from langchain.tools import tool
from config import stock_evaluations_path
import pandas as pd



def _stock_market_data(ticker_symbol: str) -> str:
    ticket_symbol = ticker_symbol.upper()
    #Sadly, API calls are only 25 per day, so will be using mocking data for this exercise:
    lower_price = random.randint(10 , 200)
    higher_price = random.randint(201 , 500)

    pe_ratio = random.randint(10 , 40)

    return f"the ticket symbol {ticket_symbol} has a lowest price of {lower_price}, and highest of {higher_price}, with a pe ratio of {pe_ratio} times per sales"


@tool(
    "stock_market_data",
    parse_docstring=True,
    description="gives you the stock market prices necessary to answer, alongside the p/e ratio of the company"
)
def stock_market_data_tool(ticker_symbol : str) -> str:
    """
    Description:
        Gets you the lowest and highest price of a stock in the last 2 years and the pe ratio

    Args:
        ticker_symbol (str): The ticker symbol to research

    Returns:
        ticker symbols highest and lowest price in the last 2 years, plus the pe ratio

    Raises:
        If there is not wnough information about the symbol and or an error in the API Call
    """

    return _stock_market_data(ticker_symbol)

def _save_stock_evals(ticket_symbol : str, LLM_Answers : list,  selected_reason : list) -> str:
    """
    Description:
        Saves the stock evals in a csv file

    Args:
        ticket_symbol (str): The ticket symbol to research
        recommendations_list (list): The list of recommendations
        price (float): The price of the stock
        p_e (float): The p/e ratio of the stock
        selected_reason (str): The reason for the recommendation

    Returns:
        Lets the user know a new sell has been done and gives the details of that new transaction

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    

    df = pd.read_csv(stock_evaluations_path)
    new_row = pd.DataFrame({
        "stock": [ticket_symbol],
        "LLM_1": [LLM_Answers[0] if len(LLM_Answers) > 0 else None],
        "LLM_2": [LLM_Answers[1] if len(LLM_Answers) > 1 else None],
        "LLM_3": [LLM_Answers[2] if len(LLM_Answers) > 2 else None],
        "LLM_4": [LLM_Answers[3] if len(LLM_Answers) > 3 else None],
        "LLM_5": [LLM_Answers[4] if len(LLM_Answers) > 4 else None],
        "one_sentence_reasoning": [selected_reason]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(stock_evaluations_path, index=False)

    return "Succcessfully saved the stock recommendation into the stock evaluations database"

def ticker_admin_tool(ticker_symbol):
    """
    A function that checks if the ticker symbol requested is in the database already or not.
    If it is, returns True, if it is not, returns False.
    """
    #Check the first column in db
    df = pd.read_csv(stock_evaluations_path)
    ticker_column  = df["stock"].values

    if ticker_symbol in ticker_column:
        print(f"The ticker symbol {ticker_symbol} its already in the db")
        return True
    else:
        print(f"The ticker symbol {ticker_symbol} it not in the DB.\nGathering info from the SEC now...")
        return False

def ticker_info_db(ticker_symbol):
    df = pd.read_csv(stock_evaluations_path)
    if df[df['stock'] == ticker_symbol].empty:
        return f"We do not have information about {ticker_symbol} in the Database. Ask user to go to the Councel of LLMs"
    else:
        ticker_row  = df[df["stock"] == ticker_symbol].to_markdown(index=False)
        
        return f"Here the info about {ticker_symbol}:\n{ticker_row}"


#Tool for Expert Financial Consultant

@tool(
    "review_stock_data",
    parse_docstring= True,
    description="gives back information about the sotck the user is asking about. If no info, lets you know next steps"
)
def review_stock_data(ticker_symbol : str) -> str:
    """
    Description:
        Gives back information about the sotck the user is asking about. If no info, lets you know next steps
    
    Args:
        ticker_symbol (str): The stock or ticker symbol of the company that will be reviewed agains users risk tolerance and portfolio.
    
    Returns:
        The ticker symbol information we have in the database. 

    Raises:
        If not info in the database, lets you know next steps.

    """
    return ticker_info_db(ticker_symbol)