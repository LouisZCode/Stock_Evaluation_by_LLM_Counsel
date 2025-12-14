
from functions import response_quaterly,response_my_portfolio, _update_portfolio_info, find_opportunities, update_risk_state
from config import portfolio_path
import pandas as pd
import gradio as gr

waiting_for_approval_state = gr.State(False)
risk_state = gr.State("")
_update_portfolio_info()

with gr.Blocks() as demo:
    with gr.Tabs():
        
        #Stock Evaluation Tab, saved into stock_evaluation.csv
        with gr.Tab("Counsel of LLMs"):
            gr.Markdown("# The Counsel that will research and categorize a stock for you...") 
            gr.ChatInterface(
                fn=response_quaterly
            )
            gr.Markdown("### NOTE: Answer based on real SEC data, but on mock stock price and P/E ratio because of API Costs.\nPlease dont use this Tech-Demo as financial advice") 

        #Manages and takes action on the current portfolio
        with gr.Tab("Trade Assistant"):
            gr.Markdown("# Your Portfolio") 
            portfolio_display = gr.DataFrame(pd.read_csv(portfolio_path))
            gr.Markdown("## Manage your Portfolio:") 
            gr.ChatInterface(
                fn=response_my_portfolio,
                additional_inputs=[waiting_for_approval_state],
                additional_outputs=[waiting_for_approval_state, portfolio_display]
            )

        with gr.Tab("Find Opportunities"):
            gr.Markdown("# Decide What to Buy.. or not to...")
            gr.Markdown("## Decide Your Risk Tolerance:") 
            risk_dropdown = gr.Dropdown(
                label="Select your Risk Tolerance",
                interactive=True,
                choices=["Y.O.L.O", "I tolerate a lot of RISK", "I tolerate little risk", "Lets take NO risks"],
                )

            risk_dropdown.change(
                fn=update_risk_state,
                inputs=risk_dropdown,
                outputs=risk_state
                )

            gr.ChatInterface(
                fn=find_opportunities,
                additional_inputs=[risk_state]
            )