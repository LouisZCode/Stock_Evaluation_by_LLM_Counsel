"""
This created the retriever for the RAG and gives a retriever_tool 
to be used by an agent

Here you find:
retriever_tool
"""

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, vector_store_path

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@tool
def retriever_tool(query: str) -> str:
    """
    Search SEC 8-Q quarterly financial filings for a company.
    Query with the ticker symbol to get financial data about earnings, revenue, growth, etc.
    """
    #Fallback in case the LLM only send the ticker symbol as the query:
    if len(query.split()) <= 5:
        query = f"{query} financial strength and earnings revenue growth"

    # Extract ticker from query (first word, uppercase)
    ticker = query.split()[0].upper()

    vector_store = FAISS.load_local(
        vector_store_path,
        embedding,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5,
        "filter" : {"ticker" : ticker}}
    )
    results = retriever.invoke(query)

    if not results:
        return "No relevant financial information found for this query."

    # Include metadata with each chunk so LLMs know the source quarter
    output = []
    for doc in results:
        ticker = doc.metadata.get('ticker', 'N/A')
        year = doc.metadata.get('year', 'N/A')
        quarter = doc.metadata.get('quarter', 'N/A')

        metadata_str = f"[{ticker} | {quarter} {year}]"
        output.append(f"{metadata_str}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)