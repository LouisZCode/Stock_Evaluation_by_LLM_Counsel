# TODO - Check the embedding and vector store creation before running again.



from config import trades_log_path, cash_log, portfolio_path, stock_evaluations_path
from functions import initialize_databases
from vector_store import download_clean_fillings




initialize_databases(trades_log_path, portfolio_path, cash_log, stock_evaluations_path)

stock = "AAPL"
result = download_clean_fillings(stock)
print(result)

"""
from UI import demo

demo.launch()
"""