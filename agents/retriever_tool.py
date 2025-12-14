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

@tool
def retriever_tool(query: str) -> str:
    """
    Search SEC 8-Q quarterly financial filings for a company.
    Query with the ticker symbol to get financial data about earnings, revenue, growth, etc.
    """
    if len(query.split()) <= 5:
        query = f"{query} financial strength and earnings revenue growth"


    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        vector_store_path,
        embedding,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)

    if not results:
        return "No relevant financial information found for this query."

    return "\n\n".join([doc.page_content for doc in results])