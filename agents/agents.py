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
explainer_prompt = prompts["EXPLAINER"]
OPPORTUNITY_FINDER_PROMPT_TEMPLATE = prompts["PORTFOLIO_RECOMMENDATOR"]


class FinancialInformation(TypedDict):
    stock: str
    financials: str
    growth: str
    lower_stock_price : str
    higher_stock_price : str
    price_description: str
    price_to_earnings: str
    recommendation: str
    reason: str

checker_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=checker_prompt,
)

openai_finance_boy = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool],
    response_format=FinancialInformation
)


anthropic_finance_boy = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool],
    response_format=FinancialInformation
)


groq_finance_boy = create_agent(
    model="groq:openai/gpt-oss-120b",
    system_prompt=quarter_results_prompt,
    checkpointer=InMemorySaver(),
    tools=[retriever_tool],
    response_format=FinancialInformation
)

simple_explaining_agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=explainer_prompt,
)


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


opportunity_agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    system_prompt=explainer_prompt,
    tools=[read_my_portfolio, review_stock_data],
    checkpointer=InMemorySaver()
)
