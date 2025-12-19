"""
Here you will find the response Logic between Gradio and the Agents for the project

You have:
response_quaterly, response_my_portfolio, find_opportunities, update_risk_state
"""


from .agents import (
    checker_agent, simple_explaining_agent, anthropic_finance_boy, mistral_finance_boy,
openai_finance_boy, opportunity_agent, my_portfolio_agent
)

import time
import pandas as pd
from config import portfolio_path
import asyncio
from langgraph.types import Command
import random


from functions import (
    ticker_admin_tool, ticker_info_db, _save_stock_evals, _update_portfolio_info,
    _extract_structured_data
    )

from vector_store import download_clean_fillings
from logs import start_new_log, log_llm_conversation


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
            result = download_clean_fillings(ticker_symbol)

            if result is None:
                yield f"No SEC filings found for '{ticker_symbol}'. This ticker may not be a US-listed company or doesn't have 10-Q filings available."
                return

            yield "Data received, now the counsel will review the data and come with a verdict, just a moment..."
            time.sleep(2)

            # Start logging for this research session
            log_file = start_new_log(ticker_symbol)

            LLM_Answers = []

            #OPENAI Research
            try:
                yield "OpenAI is thinking..."
                time.sleep(1)
                response_openai = await openai_finance_boy.ainvoke(
                    {"messages": [{"role": "user", "content": f"Analyze {ticker_symbol}'s quarterly financial performance. Look for: revenue, net income, gross margin, operational costs, cash flow, quarterly growth, total assets, and total debt"}]}
                    )
                    
                log_llm_conversation("OpenAI", response_openai, log_file)
                data_openai = _extract_structured_data(response_openai["messages"][-1].content)

                if data_openai and "financial_strenght" in data_openai:
                    LLM_Answers.append(data_openai)
                    print(f"OpenAI says: {data_openai}")
                    yield f"OpenAI says: {data_openai['financial_strenght']}\n\n"
                else:
                    print(f"OpenAI returned invalid data: {data_openai}")

                time.sleep(2)
            except Exception as e:
                print(f"OpenAI failed: {e}")

            #CLAUDE Research
            try:
                yield "Claude is thinking..."
                time.sleep(1)
                response_claude = await anthropic_finance_boy.ainvoke(
                    {"messages": [{"role": "user", "content": f"Analyze {ticker_symbol}'s quarterly financial performance. Look for: revenue, net income, gross margin, operational costs, cash flow, quarterly growth, total assets, and total debt"}]},
                )
                log_llm_conversation("Claude", response_claude, log_file)
                data_claude = _extract_structured_data(response_claude["messages"][-1].content)

                if data_claude and "financial_strenght" in data_claude:
                    LLM_Answers.append(data_claude)
                    print(f"Claude says: {data_claude}")
                    yield f"Claude says: {data_claude['financial_strenght']}\n\n"
                else:
                    print(f"Claude returned invalid data: {data_claude}")

            except Exception as e:
                print(f"Claude failed: {e}")

            #MISTRAL Research
            try:
                yield "Mistral is thinking..."
                time.sleep(1)
                response_mistral = await mistral_finance_boy.ainvoke(
                    {"messages": [{"role": "user", "content": f"Analyze {ticker_symbol}'s quarterly financial performance. Look for: revenue, net income, gross margin, operational costs, cash flow, quarterly growth, total assets, and total debt"}]},
                )
                log_llm_conversation("Mistral", response_mistral, log_file)
                data_mistral = _extract_structured_data(response_mistral["messages"][-1].content)

                if data_mistral and "financial_strenght" in data_mistral:
                    LLM_Answers.append(data_mistral)
                    print(f"Mistral says: {data_mistral}")
                    yield f"Mistral recommends: {data_mistral['financial_strenght']}\n\n"
                else:
                    print(f"Mistral returned invalid data: {data_mistral}")

            except Exception as e:
                print(f"Mistral failed: {e}") 

            if LLM_Answers:
                recommendations_list = [answer["financial_strenght"] for answer in LLM_Answers]
                selected_reason = [answer["reason"] for answer in LLM_Answers]
                _save_stock_evals(ticker_symbol, recommendations_list, selected_reason)

                yield f"Analysis complete. Results saved to database.\n\n{LLM_Answers}"

            else:
                yield "All LLM calls failed. Please try again later."
                print("all calls to LLMs failed. Try again later")
            

   
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