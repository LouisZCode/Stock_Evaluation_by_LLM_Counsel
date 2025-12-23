"""
All the agents coming from create_agent will be here:

You can find the following agents:
checker_agent, openai_finance_boy, anthropic_finance_boy, google_finance_boy,
simple_explaining_agent, my_portfolio_agent, opportunity_agent
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
from langchain.agents.middleware import HumanInTheLoopMiddleware

from dotenv import load_dotenv
from .prompt_loader import load_prompts
from .retriever_tool import retriever_tool

from functions import (
    read_my_portfolio,add_to_portfolio,remove_from_portfolio, add_cash_tool, withdraw_cash_tool,
    cash_position_count, review_stock_data
    )

load_dotenv()

prompts = load_prompts()
quarter_results_prompt = prompts["QUATERLY_RESULTS_EXPERT"]
my_portfolio_prompt = prompts["MY_PORTFOLIO_EXPERT"]
checker_prompt = prompts["CHECKER"]
counsel_voice = prompts["THE_COUNSEL_VOICE"]
debater_prompt = prompts["THE_DEBATER"]
explainer_prompt = prompts["EXPLAINER"]
OPPORTUNITY_FINDER_PROMPT_TEMPLATE = prompts["PORTFOLIO_RECOMMENDATOR"]

# Answer format from the Finance Boys, so it is heterogeneous
class FinancialInformation(TypedDict):
    stock: str
    revenue: str
    revenue_reason: str
    net_income: str
    net_income_reason: str
    gross_margin: str
    gross_margin_reason: str
    operational_costs: str
    operational_costs_reason: str
    cash_flow: str
    cash_flow_reason: str
    quaterly_growth: str
    quaterly_growth_reason: str
    total_assets: str
    total_assets_reason: str
    total_debt: str
    total_debt_reason: str
    financial_strenght: str
    overall_summary: str

"""
Cleans the Human Query so it gets the Ticker Symbol, and passes it on to the Finance Boys
"""
checker_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=counsel_voice,
    checkpointer=InMemorySaver()
)

"""
### Same Agent, Different LLMS, get Data from Database and gives an answer in RAG
"""
openai_finance_boy = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=quarter_results_prompt,
    tools=[retriever_tool],
    response_format=FinancialInformation
)

anthropic_finance_boy = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=quarter_results_prompt,
    tools=[retriever_tool],
    response_format=FinancialInformation
)

mistral_finance_boy = create_agent(
    model="mistral-large-2512",
    system_prompt=quarter_results_prompt,
    tools=[retriever_tool],
    response_format=FinancialInformation
)



"""
LLMs that take thee information from the finance boys, and if needed, talk about it with each other to reach a financial "truth". The Agora
"""

openai_socrates = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=debater_prompt,
    checkpointer=InMemorySaver()
)

anthropic_pythagoras = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=debater_prompt,
    checkpointer=InMemorySaver()
)

mistral_diogenes = create_agent(
    model="mistral-large-2512",
    system_prompt=debater_prompt,
    checkpointer=InMemorySaver()
)


"""
Adminitrative AI, that can get cash, buy and sell stocks in your name, etc..
"""
my_portfolio_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=my_portfolio_prompt,
    checkpointer=InMemorySaver(),
    tools=[read_my_portfolio, add_to_portfolio, remove_from_portfolio, add_cash_tool, withdraw_cash_tool, cash_position_count],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "add_to_portfolio": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Confirm addition of new stock to portfolio"},

                "remove_from_portfolio" :{
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Confirm removal of a stock from portfolio"}
                }
            )
        ])




"""
AI to discuss opportunities based on your risk tolerance.
"""
opportunity_agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=explainer_prompt,
    tools=[read_my_portfolio, review_stock_data],
    checkpointer=InMemorySaver()
)
