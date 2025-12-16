"""
This created the retriever for the RAG and gives a retriever_tool
to be used by an agent

Here you find:
retriever_tool
"""

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, vector_store_path, ticker_quarters_path, bm25_chunks_path
import json

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_ticker_quarters():
    """Load the ticker quarters mapping from JSON file."""
    if ticker_quarters_path.exists():
        with open(ticker_quarters_path, "r") as f:
            return json.load(f)
    return {}


def get_bm25_results(query: str, ticker: str, k: int = 5) -> list:
    """Get BM25 keyword search results filtered by ticker."""
    if not bm25_chunks_path.exists():
        return []

    with open(bm25_chunks_path, "r") as f:
        all_chunks = json.load(f)

    # Filter by ticker first (faster search)
    ticker_chunks = [
        Document(page_content=c["content"], metadata=c["metadata"])
        for c in all_chunks if c["metadata"]["ticker"] == ticker
    ]

    if not ticker_chunks:
        return []

    # Create BM25 retriever and search
    bm25 = BM25Retriever.from_documents(ticker_chunks)
    bm25.k = k
    return bm25.invoke(query)


@tool
def retriever_tool(query: str) -> str:
    """
    Search SEC 8-Q quarterly financial filings for a company.
    Query with the ticker symbol to get financial data about earnings, revenue, growth, etc.
    """
    # Fallback in case the LLM only sends the ticker symbol as the query:
    if len(query.split()) <= 5:
        query = f"{query} financial strength and earnings revenue growth"

    # Extract ticker from query (first word, uppercase)
    ticker = query.split()[0].upper()

    vector_store = FAISS.load_local(
        vector_store_path,
        embedding,
        allow_dangerous_deserialization=True
    )

    results = []

    # Part A: Get 2 chunks from each of the latest 3 quarters (6 total)
    quarters_data = load_ticker_quarters()
    latest_quarters = quarters_data.get(ticker, [])[:3]  # Top 3 quarters

    for year, quarter in latest_quarters:
        quarter_retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 2,
                "filter": {"ticker": ticker, "year": year, "quarter": quarter}
            }
        )
        results.extend(quarter_retriever.invoke(query))

    # Part B: Get 9 more chunks from general semantic search (ticker-filtered only)
    general_retriever = vector_store.as_retriever(
        search_kwargs={"k": 9, "filter": {"ticker": ticker}}
    )
    results.extend(general_retriever.invoke(query))

    # Part C: Get 5 chunks from BM25 keyword search
    bm25_results = get_bm25_results(query, ticker, k=5)
    results.extend(bm25_results)

    # Dedupe by content (in case of overlap between semantic and BM25 search)
    seen = set()
    unique_results = []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append(doc)

    if not unique_results:
        return "No relevant financial information found for this query."

    # Include metadata with each chunk so LLMs know the source quarter
    output = []
    for doc in unique_results:
        doc_ticker = doc.metadata.get('ticker', 'N/A')
        year = doc.metadata.get('year', 'N/A')
        quarter = doc.metadata.get('quarter', 'N/A')

        metadata_str = f"[{doc_ticker} | {quarter} {year}]"
        output.append(f"{metadata_str}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)