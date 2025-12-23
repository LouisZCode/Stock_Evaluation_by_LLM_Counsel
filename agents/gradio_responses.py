"""
Here you will find the response Logic between Gradio and the Agents for the project

You have:
response_quaterly, response_my_portfolio, find_opportunities, update_risk_state
"""


from .agents import (
    checker_agent, anthropic_finance_boy, mistral_finance_boy,
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
from logs import start_new_log, log_llm_conversation, log_debate_check, log_llm_timing, log_harmonization
from functions import (
    calculate_agreement, get_metric_comparison, fill_missing_with_consensus,
    recalculate_strength_scores, calculate_agreement_from_scores, harmonize_and_check_debates
)


"""
Agent that reads the vector stores, and gives you info about the quaterly information
"""
async def response_quaterly(message, history):

    check_ticker = checker_agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable" : {"thread_id" : "002"}}
    )

    print("before IF statements:")
    print(check_ticker["messages"][-1].content)
    print()

    msg = check_ticker["messages"][-1].content
    ticker_symbol = msg.split()[-1]

    if not check_ticker["messages"][-1].content == f"We will research the ticker {ticker_symbol}":
        print(check_ticker["messages"][-1].content)
        yield check_ticker["messages"][-1].content
        return
    
    else:
        
        if ticker_admin_tool(ticker_symbol):
            yield "The councel already had researched this Ticker, gathering the info form the database..."
            time.sleep(2)
            #agent that explains the info:
            ticker_info = ticker_info_db(ticker_symbol)

            explainer_agent = checker_agent.invoke(
            {"messages": [{"role": "user", "content": f"{ticker_info}"}]},
            {"configurable" : {"thread_id" : "002"}}
            )
            explainer_agent["messages"][-1].content

            yield explainer_agent["messages"][-1].content
            return
        
        else:
            yield f"We will research the ticker {ticker_symbol}"
            await asyncio.sleep(2)
            yield "The Counsel is gathering data of this company from the SEC directly, this will take 1 minute..."
            await asyncio.sleep(1)
            result = download_clean_fillings(ticker_symbol)

            if result is None:
                yield f"No SEC filings found for '{ticker_symbol}'. This ticker may not be a US-listed company or doesn't have 10-Q filings available."
                return

            yield "Data received, now the counsel will review the data and come with a verdict, just a moment..."
            time.sleep(2)

            # Start logging for this research session
            log_file = start_new_log(ticker_symbol)

            yield "The Counsel is deliberating on the financial data..."

            # LLM configuration for parallel calls
            LLM_CONFIGS = [
                ("OpenAI", openai_finance_boy),
                ("Claude", anthropic_finance_boy),
                ("Mistral", mistral_finance_boy),
            ]

            # Build the prompt once (same for all LLMs)
            prompt = f"Analyze {ticker_symbol}'s quarterly financial performance. Look for: revenue, net income, gross margin, operational costs, cash flow, quarterly growth, total assets, and total debt"

            # Create tasks for parallel execution
            tasks = [
                llm.ainvoke({"messages": [{"role": "user", "content": prompt}]})
                for name, llm in LLM_CONFIGS
            ]

            # Run all LLM calls in parallel
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed_time = time.time() - start_time

            # Log timing
            print(f"All LLMs responded in {elapsed_time:.2f}s (parallel)")
            log_llm_timing(elapsed_time, log_file)

            # Process responses after all complete
            LLM_Answers = []

            for (name, _), response in zip(LLM_CONFIGS, responses):
                # Check if this LLM failed
                if isinstance(response, Exception):
                    print(f"{name} failed: {response}")
                    continue

                # Log the conversation
                log_llm_conversation(name, response, log_file)

                # Extract structured data
                data = _extract_structured_data(response["messages"][-1].content)

                if data and "financial_strenght" in data:
                    LLM_Answers.append(data)
                    print(f"{name} says: {data}")
                    yield f"{name} says: {data['financial_strenght']}\n\n"
                else:
                    print(f"{name} returned invalid data: {data}")

            # Check agreement between LLMs and log debate status
            if len(LLM_Answers) >= 2:
                # Original agreement and comparison (from LLM responses)
                original_agreement = calculate_agreement(LLM_Answers)
                original_comparison = get_metric_comparison(LLM_Answers)

                # Fill missing values where consensus exists
                filled_analyses = fill_missing_with_consensus(LLM_Answers)

                # Harmonize same-tier ratings and identify metrics that need debate
                harmonize_result = harmonize_and_check_debates(filled_analyses)
                harmonized_analyses = harmonize_result['harmonized_analyses']
                metrics_to_debate = harmonize_result['metrics_to_debate']

                # Recalculate scores after harmonization
                recalc_scores = recalculate_strength_scores(harmonized_analyses)

                # Get comparison table after filling (for original log)
                filled_comparison = get_metric_comparison(filled_analyses)

                # Log original debate check (for comparison)
                log_debate_check(original_agreement, None, original_comparison, filled_comparison, log_file)

                # Log harmonization results with new format
                log_harmonization(harmonize_result, recalc_scores, log_file)

                # Use tier-based decision for debate
                if metrics_to_debate:
                    yield f"Debate triggered on: {', '.join(metrics_to_debate)}\n\n"

                # Use harmonized analyses for the rest of the flow
                LLM_Answers = harmonized_analyses

            if LLM_Answers:
                recommendations_list = [answer["financial_strenght"] for answer in LLM_Answers]
                selected_reason = [answer["overall_summary"] for answer in LLM_Answers]
                _save_stock_evals(ticker_symbol, recommendations_list, selected_reason)

                check_ticker = checker_agent.invoke(
                    {"messages": [{"role": "user", "content": f"[FROM THE COUNSEL]: Here are the financial findings for {ticker_symbol}:\n{LLM_Answers}"}]},
                    {"configurable" : {"thread_id" : "002"}}
                )
                yield check_ticker["messages"][-1].content
                return 

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