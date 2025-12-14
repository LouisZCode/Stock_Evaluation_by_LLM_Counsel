from config import trades_log_path, cash_log, portfolio_path, stock_evaluations_path, database_path
from functions import initialize_databases


initialize_databases(database_path, trades_log_path, portfolio_path, cash_log, stock_evaluations_path)

from UI import demo
demo.launch()