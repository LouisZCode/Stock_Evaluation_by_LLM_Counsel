# Todo List - Stock Evaluation by LLM Counsel

## STOCK-MARKET-DATA
**Situation:**
The stock market data for now is fake, as the API call from https://www.alphavantage.co/ its expensive.  
Compare it also against https://finnhub.io/

**Proposal-1**
Implement real data well curated and inject it in all 3 LLMs, meaning, 1 API call instead of 3, reducing costs.

## RAG-001: Cross-ticker contamination in semantic search

**Situation:**
When searching for financial data (e.g., "ORCL total revenue"), the FAISS semantic search returns documents from other tickers (GOOGL, MSFT, etc.) because the embedding model doesn't understand ticker symbols as identifiers - it treats them as regular words and matches based on semantic similarity of the surrounding text.

**Proposal-1:**
Add metadata filtering to the retriever. Extract the ticker symbol from the query and pass it as a filter to FAISS before semantic search runs. This ensures only documents with matching ticker metadata are searched.

**Proposal-2:**
Implement hybrid search (BM25 + semantic). BM25 does keyword matching which would catch exact ticker matches, combined with semantic search for understanding financial concepts.

---

## RAG-002: Boilerplate text returned instead of actual financial data

**Situation:**
The retriever often returns section headers like "Total Revenues and Operating Expenses" or generic boilerplate like "fluctuations in our financial results" instead of chunks containing actual numbers (revenue figures, net income, growth percentages).

**Proposal-1:**
Add boilerplate filtering during ingestion in `vs_addition.py`. Skip or flag chunks that are too short, contain only headers, or match known boilerplate patterns.

**Proposal-2:**
Implement BM25 hybrid search. Keyword search would prioritize chunks containing actual numbers and financial terms like "$", "million", "billion", "increased", "decreased".

**Proposal-3:**
Improve chunking strategy - use larger chunks or semantic chunking to keep financial tables and their context together.

---

## COUNSEL-001: Random selection for final verdict

**Situation:**
When multiple LLMs return their recommendations, the final "reason" shown to users is randomly selected from the list (`random.choice(reasons_list)`). This doesn't synthesize the different perspectives or handle disagreements intelligently.

**Proposal-1:**
Use a "smarter" LLM to read all the answers and synthesize a final verdict. Pass all recommendations and reasons to a summarizer agent that weighs the evidence and produces a coherent explanation.

**Proposal-2:**
Implement weighted voting based on confidence scores or data quality (e.g., if one LLM got better retrieval results, weight its opinion higher).

---

## PORTFOLIO-001: Approval message lacks transaction details

**Situation:**
When the portfolio agent triggers human-in-the-loop approval for buy/sell operations, the approval message is generic ("The agent wants BUY or SELL stock"). It doesn't show the specific stock, quantity, or price from the `__interrupt__` value.

**Proposal-1:**
Extract transaction details from `response["__interrupt__"]` and populate the approval message with: stock symbol, action (BUY/SELL), quantity, and price.

---

## INGEST-001: Missing ticker causes TypeError

**Situation:**
When a user requests research on a ticker that doesn't exist in SEC EDGAR (e.g., typo or delisted company), the `download_clean_fillings()` function fails with `TypeError: 'NoneType' has no len()` because `company.get_filings()` returns None.

**Proposal-1:**
Add try/except around the filings fetch. Return a user-friendly error message like "Ticker not found in SEC database" instead of crashing.

**Proposal-2:**
Add ticker validation before calling SEC API - check against a known list of valid tickers or use a ticker lookup service.

---
