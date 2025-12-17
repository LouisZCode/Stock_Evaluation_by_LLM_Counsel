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


def get_bm25_results(query: str, ticker: str, k: int = 5, quarters_filter: list = None, exclude: bool = False) -> list:
    """
    Get BM25 keyword search results filtered by ticker and optionally by quarters.

    Args:
        query: Search query
        ticker: Stock ticker symbol
        k: Number of results to return
        quarters_filter: List of (year, quarter) tuples to filter by
        exclude: If True, exclude the quarters_filter instead of including them
    """
    if not bm25_chunks_path.exists():
        return []

    with open(bm25_chunks_path, "r") as f:
        all_chunks = json.load(f)

    # Filter by ticker first
    ticker_chunks = [c for c in all_chunks if c["metadata"]["ticker"] == ticker]

    # Apply quarters filter if provided
    if quarters_filter:
        quarters_set = set((y, q) for y, q in quarters_filter)
        if exclude:
            # Exclude these quarters (for older data)
            ticker_chunks = [c for c in ticker_chunks
                           if (c["metadata"]["year"], c["metadata"]["quarter"]) not in quarters_set]
        else:
            # Include only these quarters (for recent data)
            ticker_chunks = [c for c in ticker_chunks
                           if (c["metadata"]["year"], c["metadata"]["quarter"]) in quarters_set]

    # Convert to Documents
    docs = [Document(page_content=c["content"], metadata=c["metadata"]) for c in ticker_chunks]

    if not docs:
        return []

    # Create BM25 retriever and search
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25.invoke(query)


def dedupe_chunks(chunks: list, limit: int = None) -> list:
    """Remove duplicate chunks by content, optionally limiting to first N unique."""
    seen = set()
    unique = []
    for doc in chunks:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
            if limit and len(unique) >= limit:
                break
    return unique


@tool
def retriever_tool(query: str) -> str:
    """
    Search SEC 10-Q quarterly financial filings for a company.
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

    # Load quarters data
    quarters_data = load_ticker_quarters()
    all_quarters = quarters_data.get(ticker, [])
    latest_quarters = all_quarters[:3]  # Most recent 3 quarters
    older_quarters = all_quarters[3:]   # Everything else

    print(f"\n[DEBUG] Ticker: {ticker}")
    print(f"[DEBUG] All quarters: {all_quarters}")
    print(f"[DEBUG] Latest 3: {latest_quarters}")
    print(f"[DEBUG] Older: {older_quarters}")

    # ============================================================
    # Part A: Latest 3 quarters (semantic + BM25 → 10 unique chunks)
    # ============================================================
    part_a_results = []

    # Semantic search on latest quarters
    # fetch_k: get more candidates before filtering to ensure we find matches
    for year, quarter in latest_quarters:
        quarter_retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
                "fetch_k": 500,
                "filter": {"ticker": ticker, "year": year, "quarter": quarter}
            }
        )
        part_a_results.extend(quarter_retriever.invoke(query))

    # BM25 search on latest quarters
    bm25_recent = get_bm25_results(query, ticker, k=10, quarters_filter=latest_quarters, exclude=False)
    part_a_results.extend(bm25_recent)

    # Dedupe to 10 unique chunks
    part_a_unique = dedupe_chunks(part_a_results, limit=10)

    # ============================================================
    # Part B: Older quarters (semantic + BM25 → 10 unique chunks)
    # ============================================================
    part_b_results = []

    print(f"[DEBUG] Part A results count: {len(part_a_unique)}")

    if older_quarters:
        print(f"[DEBUG] Processing Part B with {len(older_quarters)} older quarters")
        # Semantic search on older quarters
        # fetch_k: get more candidates before filtering to ensure we find matches
        semantic_older_count = 0
        for year, quarter in older_quarters[:5]:  # Limit to 5 older quarters for efficiency
            quarter_retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                    "fetch_k": 500,
                    "filter": {"ticker": ticker, "year": year, "quarter": quarter}
                }
            )
            results = quarter_retriever.invoke(query)
            print(f"[DEBUG] Semantic search {quarter} {year}: {len(results)} results")
            semantic_older_count += len(results)
            part_b_results.extend(results)

        print(f"[DEBUG] Part B semantic total: {semantic_older_count}")

        # BM25 search on older quarters (exclude latest 3)
        bm25_older = get_bm25_results(query, ticker, k=10, quarters_filter=latest_quarters, exclude=True)
        print(f"[DEBUG] Part B BM25 results: {len(bm25_older)}")
        part_b_results.extend(bm25_older)

        print(f"[DEBUG] Part B total before dedupe: {len(part_b_results)}")

        # Dedupe to 10 unique chunks
        part_b_unique = dedupe_chunks(part_b_results, limit=10)
        print(f"[DEBUG] Part B after dedupe: {len(part_b_unique)}")
    else:
        part_b_unique = []

    # ============================================================
    # Combine: 50% recent + 50% historical = 20 chunks max
    # ============================================================
    final_results = part_a_unique + part_b_unique
    print(f"[DEBUG] Final results: {len(final_results)} (Part A: {len(part_a_unique)}, Part B: {len(part_b_unique)})")

    if not final_results:
        return "No relevant financial information found for this query."

    # Include metadata with each chunk so LLMs know the source quarter
    output = []
    for doc in final_results:
        doc_ticker = doc.metadata.get('ticker', 'N/A')
        year = doc.metadata.get('year', 'N/A')
        quarter = doc.metadata.get('quarter', 'N/A')

        metadata_str = f"[{doc_ticker} | {quarter} {year}]"
        output.append(f"{metadata_str}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)