"""
Here you will find the response Logic between Gradio and the Agents for the project

You have:
response_quaterly, response_my_portfolio, find_opportunities, update_risk_state
"""


from agents import (
    checker_agent, simple_explaining_agent, anthropic_finance_boy, groq_finance_boy,
openai_finance_boy, opportunity_agent, my_portfolio_agent
)

import time
import pandas as pd
from config import portfolio_path
import asyncio
from langgraph.types import Command
import random


from functions import (
    ticker_admin_tool, _stock_market_data, ticker_info_db, _save_stock_evals, _update_portfolio_info, 
    _extract_structured_data
    )

from vector_store import download_clean_fillings


"""
Agent that reads the vector stores, and gives you info about the quaterly information
"""
async def response_quaterly(message, history):

    check_ticker = checker_agent.invoke(
        {"messages": [{"role": "user", "content": message}]}
    )

    print("before IF statements:")
    print(check_ticker["messages"][-1].content)
    print()

    if check_ticker["messages"][-1].content == 'No clear symbol or company mentioned, could you try again please?':
        print(check_ticker["messages"][-1].content)
        yield "No clear symbol or company mentioned, could you try again please?"
        return 
    
    else:
        ticker_symbol = check_ticker["messages"][-1].content
        yield f"I have found the ticker symbol {ticker_symbol} in the users query, thinking..."
        time.sleep(1)

        if ticker_admin_tool(ticker_symbol):
            yield "The councel already had researched this Ticker, gathering the info form the database..."
            time.sleep(2)
            #agent that explains the info:
            ticker_info = ticker_info_db(ticker_symbol)

            explainer_agent = simple_explaining_agent.invoke(
            {"messages": [{"role": "user", "content": f"{ticker_info}"}]}
            )
            explainer_agent["messages"][-1].content

            yield explainer_agent["messages"][-1].content
            return
        
        else:

            yield "Getting data for this company from the SEC directly, this will take 1 minute..."
            await asyncio.sleep(1)
            download_clean_fillings(ticker_symbol)
            
            #OPENAI Research
            yield "Data received, now the councel with review the data and come with a veridict, just a moment..."
            time.sleep(2)

            prices_pe_data = _stock_market_data(ticker_symbol)

            response_openai = await openai_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": f"Research {ticker_symbol}, more info: {prices_pe_data}"}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_openai = _extract_structured_data(response_openai["messages"][-1].content)
            #print(f"OpenAi Says:{data_openai}")
            yield f"OpenAI recommends: {data_openai["recommendation"]}\n\n"
            time.sleep(2)


            #CLAUDE Research
            response_claude = await anthropic_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": f"Research {ticker_symbol}, more info: {prices_pe_data}"}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_claude = _extract_structured_data(response_claude["messages"][-1].content)
            #print(f"Claude says: {data_claude}")
            yield f"Claude recommends: {data_claude["recommendation"]}\n\n"


            #Groq Research
            response_groq = await groq_finance_boy.ainvoke(
                {"messages": [{"role": "user", "content": f"Research {ticker_symbol}, more info: {prices_pe_data}"}]},
                {"configurable": {"thread_id": "thread_001"}}
            )
            data_groq = _extract_structured_data(response_groq["messages"][-1].content)
            #print(f"Google says: {data_claude}")
            yield f"Groq recommends: {data_claude["recommendation"]}\n\n"


            AI1 = data_openai["recommendation"]
            AI2 = data_claude["recommendation"]
            AI3 = data_groq["recommendation"]

            AI1_reason = data_openai["reason"]
            AI2_reason = data_claude["reason"]
            AI3_reason = data_groq["reason"]

            AI1_price = data_openai["higher_stock_price"]
            AI2_price = data_claude["higher_stock_price"]
            AI3_price = data_groq["higher_stock_price"]

            AI1_pri_de = data_openai["price_description"]
            AI2_pri_de = data_claude["price_description"]
            AI3_pri_de = data_groq["price_description"]


            AI1_pe = data_openai["price_to_earnings"]
            AI2_pe = data_claude["price_to_earnings"]
            AI3_pe = data_groq["price_to_earnings"]


            recommendations_list = [AI1, AI2, AI3]
            reasons_list = [AI1_reason, AI2_reason, AI3_reason]
            price_list = [AI1_price, AI2_price, AI3_price]
            price = random.choice(price_list)
            p_e_list = [AI1_pe, AI2_pe, AI3_pe]
            p_e = random.choice(p_e_list)
            price_des_list = [AI1_pri_de, AI2_pri_de, AI3_pri_de]
            price_description = random.choice(price_des_list)

            selected_reason = random.choice(reasons_list)

            ticket_symbol = data_openai["stock"]

            saved_database = _save_stock_evals(ticket_symbol, AI1, AI2, AI3, price, price_description,  p_e, selected_reason)

            if recommendations_list.count("Buy") >= 2:
                yield f"The councel of LLMS recommends to BUY this stock, the reason:\n\n{selected_reason}\n\n{saved_database}"

            elif recommendations_list.count("Sell") >= 2:
                yield f"The councel of LLMS recommends to SELL this stock, the reason:\n{selected_reason}\n\n{saved_database}"

            else:
                yield f"The councel of LLMS recommends to HOLD this stock, the reason:\n{selected_reason}\n\n{saved_database}"
            

   
"""
Agent that manages the portfolio
"""
def response_my_portfolio(message, history, waiting_for_approval):

    if waiting_for_approval:

        decision = message.lower().strip()

        if decision in ["yes" , "y", "approve", "buy", "sell"]:
            decision = "approve"
        elif decision in ["no", "n", "reject"]:
            decision = "reject"
        else:
            decision = "reject"

        response = my_portfolio_agent.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            {"configurable" : {"thread_id" : "thread_002"}}
            )

        
        for i, msg in enumerate(response["messages"]):
            msg.pretty_print()

        return response["messages"][-1].content, False, pd.read_csv(portfolio_path)

    else:

        messages = history + [{"role": "user", "content": message}]

        response = my_portfolio_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": "thread_002"}}
        )

        for i, msg in enumerate(response["messages"]):
            msg.pretty_print()

        if "__interrupt__" in response:

            approval_message = (
                f"⚠️ **Approval Required** ⚠️\n\n"
                f"The agent wants BUY or SELL stock\n"
                f"Do you approve? (yes/no)"
                )

            return approval_message, True, pd.read_csv(portfolio_path)

        return response["messages"][-1].content, False, pd.read_csv(portfolio_path)


# TODO grab the data in the interrupt__ value, and use it to selfpopulate correctly aproval_message
# with the stock, if it is BUY or SELL, quantity and price

    
async def find_opportunities(message, history, risk_state):

    response = await opportunity_agent.ainvoke(
        {"messages": [{"role": "user", "content": f"my risk restuls are: {risk_state}, and here is my query: {message}"}]},
        config={"configurable": {"thread_id": "opp_thread_1"}}
    )

    return response["messages"][-1].content


def update_risk_state(risk_value):
    print(f"Model changed to: {risk_value}")
    
    if risk_value == "Y.O.L.O":
        risk_value = "I prefer you to Identify high-volatility, speculative micro-cap stocks with massive upside potential. Ignore standard safety metrics. Focus on aggressive growth narratives."
    if risk_value =="I tolerate a lot of RISK":
        risk_value ="I prefer you to Focus on growth stocks with high beta. Accept significant volatility for the chance of market-beating returns."
    if risk_value =="I tolerate little risk":
        risk_value ="I prefer you to Balance growth and stability. Look for established companies with decent growth prospects and reasonable valuations."
    if risk_value =="Lets take NO risks":
        risk_value ="I prefer you to Prioritize capital preservation and steady income. Focus on blue-chip, dividend-paying aristocrats with low volatility."


    return risk_value  # Return new state value