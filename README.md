# Stock Evaluation by LLM Counsel

An intelligent investment advisory platform that synthesizes opinions from multiple Large Language Models (LLMs) using real SEC quarterly financial data to evaluate stocks and manage investment portfolios.

## Overview

This project creates a "Council of LLMs" - multiple AI models that independently analyze SEC 10-Q filings and provide stock recommendations. By combining perspectives from OpenAI, Anthropic Claude, and Mistral, the system reduces single-model bias and provides more balanced investment insights.

### Key Features

- **Multi-LLM Consensus**: Three independent LLMs analyze the same financial data and vote on recommendations (Buy/Hold/Sell)
- **Real SEC Data**: Downloads and processes 2 years of 10-Q quarterly filings (8 reports) from the SEC EDGAR database
- **RAG-Powered Analysis**: Uses vector store retrieval with metadata filtering and quarter-priority search to ground LLM responses in real financial documents
- **Portfolio Management**: Track holdings, execute trades, and manage cash with human-in-the-loop approval
- **Risk-Aligned Recommendations**: Get investment suggestions tailored to your risk tolerance

## Architecture

```
+------------------------------------------------------------------+
|                          GRADIO UI                               |
|  +------------------+ +------------------+ +------------------+   |
|  | Council of LLMs  | | Trade Assistant  | | Find Opportun.   |  |
|  +--------+---------+ +--------+---------+ +--------+---------+  |
+-----------|-------------------|--------------------|-------------+
            |                   |                    |
            v                   v                    v
+------------------------------------------------------------------+
|                           AGENTS                                 |
|  +----------+ +----------+ +----------+ +--------------------+   |
|  |  OpenAI  | |  Claude  | | Mistral  | |  Portfolio Agent   |   |
|  +----+-----+ +----+-----+ +----+-----+ +---------+----------+   |
+-------|------------|------------|-----------------|---------------+
        |            |            |                 |
        v            v            v                 v
+------------------------------------------------------------------+
|                     VECTOR STORE (FAISS)                         |
|               SEC 10-Q Filings -> Embeddings -> Search           |
+------------------------------------------------------------------+
```

## Project Structure

```
Stock_Evaluation_by_LLM_Counsel/
|-- agents/                     # LLM agent definitions
|   |-- agents.py               # Agent configurations (OpenAI, Claude, Mistral)
|   |-- prompts.yaml            # System prompts for all agents
|   |-- retriever_tool.py       # RAG retrieval tool
|   |-- gradio_responses.py     # UI response handlers
|
|-- config/                     # Configuration
|   |-- constants.py            # API keys, embedding model
|   |-- paths.py                # Database file paths
|
|-- functions/                  # Core business logic
|   |-- agent_tools_cash.py     # Cash management tools
|   |-- agent_tools_portfolio.py # Portfolio tools
|   |-- stock_data_management.py # Stock evaluation DB
|   |-- initialize_databases.py  # CSV database setup
|
|-- vector_store/               # SEC data processing
|   |-- vs_addition.py          # Download & embed SEC filings
|
|-- UI/                         # User interface
|   |-- gradio.py               # Gradio web app
|
|-- market_data/                # Stock price data
|-- logs/                       # LLM conversation logs (gitignored)
|-- data/                       # Generated databases (gitignored)
|   |-- csv/                    # Portfolio, trades, evaluations
|   |-- vector_store/           # FAISS index + ticker_quarters.json
|
|-- tests/                      # Test files
|-- main.py                     # Application entry point
|-- pyproject.toml              # Dependencies
```

## Installation

### Prerequisites

- Python 3.12+
- API keys for: OpenAI, Anthropic, Mistral

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Stock_Evaluation_by_LLM_Counsel.git
   cd Stock_Evaluation_by_LLM_Counsel
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   # Or using uv:
   uv sync
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   MISTRAL_API_KEY=your_mistral_key
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Open in browser**

   Navigate to `http://127.0.0.1:7860`

## Usage

### Tab 1: Council of LLMs

Ask about any stock and receive analysis from three independent LLMs:

```
User: "What do you think about Tesla?"

-> Downloads 2 years of SEC 10-Q filings (8 quarters)
-> Creates/updates vector store
-> Each LLM queries the data independently
-> Returns consensus recommendation (Buy/Hold/Sell)
```

### Tab 2: Trade Assistant

Manage your portfolio with natural language:

```
User: "Buy 10 shares of NVDA at $450"
Agent: "Approval Required - Do you approve? (yes/no)"
User: "yes"
Agent: "Successfully purchased 10 shares of NVDA"
```

### Tab 3: Find Opportunities

Get personalized recommendations based on your risk tolerance:

- Y.O.L.O - High-volatility speculative picks
- High Risk - Growth stocks with significant upside
- Low Risk - Balanced growth and stability
- No Risk - Blue-chip dividend aristocrats

## How It Works

### 1. Data Ingestion

When you query a new stock:
1. Downloads latest 8 10-Q filings from SEC EDGAR (2 years of quarterly data)
2. Parses HTML into text chunks (1000 chars, 100 overlap)
3. Adds metadata: ticker, year, quarter, filing date
4. Generates embeddings using `all-MiniLM-L6-v2`
5. Stores in FAISS vector database
6. Updates `ticker_quarters.json` with available quarters for fast lookup

### 2. Analysis

Each LLM independently:
1. Receives the query with stock price context
2. Searches the vector store with smart retrieval:
   - 6 chunks guaranteed from latest 3 quarters (2 per quarter)
   - 9 chunks from semantic search (best matches)
   - Metadata filtering ensures correct ticker (no cross-contamination)
3. Analyzes financials, growth, valuation
4. Returns structured recommendation

### 3. Consensus

The system aggregates responses:
- 2+ "Buy" votes -> **BUY** recommendation
- 2+ "Sell" votes -> **SELL** recommendation
- Otherwise -> **HOLD** recommendation

## Configuration

### Adding/Changing LLMs

Edit `agents/agents.py` to add new models:

```python
new_finance_boy = create_agent(
    model="provider:model-name",
    system_prompt=quarter_results_prompt,
    tools=[retriever_tool],
    response_format=FinancialInformation
)
```

### Adjusting Retrieval

Edit `agents/retriever_tool.py`:

```python
# Change number of chunks retrieved
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM Framework | LangChain, LangGraph |
| LLM Providers | OpenAI, Anthropic, Mistral |
| Vector Store | FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| SEC Data | edgartools |
| Web UI | Gradio |
| Data Storage | CSV (Pandas) |

## Limitations

- **Stock Prices**: Currently uses mock/random price data (API rate limits)
- **Real-time Data**: SEC filings are quarterly, not real-time
- **Not Financial Advice**: This is an educational project, not investment advice

## Future Development

- [x] Metadata filtering to prevent cross-ticker contamination
- [x] Quarter-priority retrieval (latest 3 quarters guaranteed)
- [ ] Boilerplate filtering to remove SEC headers/generic text
- [ ] BM25 hybrid search for better keyword matching
- [ ] Smarter consensus logic (synthesize LLM responses vs random selection)
- [ ] Real stock price API integration
- [ ] Historical performance tracking
- [ ] Portfolio analytics dashboard
- [ ] Support for 10-K annual reports

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Disclaimer

This application is for educational and research purposes only. It does not constitute financial advice. Always consult with a qualified financial advisor before making investment decisions.
