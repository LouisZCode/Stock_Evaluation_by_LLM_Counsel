"""
The tool used to get clean data from the SEC official website and add it to the vector store.
Creates a new vector store if none exists, or appends to the existing one.

You can find:
download_clean_fillings
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from edgar import Company, set_identity
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from config import SEC_IDENTITY, EMBEDDING_MODEL, vector_store_path, ticker_quarters_path, bm25_chunks_path

from datetime import datetime
import json

set_identity(SEC_IDENTITY)

# Boilerplate patterns to filter out during ingestion (RAG-004)
# These are SEC form templates with zero financial value
BOILERPLATE_BLOCKLIST = [
    "Indicate by check mark",
    "Large accelerated filer",
    "Smaller reporting company",
    "Emerging growth company",
    "filed all reports required to be filed",
    "Securities Exchange Act of 1934",
    "incorporated by reference",
    "www.sec.gov",
    "I.R.S. Employer Identification",
    "internal control over financial reporting",
    "Principal Accounting Officer",
    "Principal Financial Officer",
    "SIGNATURES",
    "Pursuant to the requirements of",
    "Exhibit Number",
    "Exhibit 31",
    "Exhibit 32",
    "forward-looking statements",
]


def is_boilerplate(text: str) -> bool:
    """Check if chunk contains SEC form boilerplate."""
    text_lower = text.lower()
    return any(pattern.lower() in text_lower for pattern in BOILERPLATE_BLOCKLIST)


def download_clean_fillings(ticker, keep_files=False): # <--- Added flag
    """
    Gets filings, processes them, and saves to VectorStore.
    keep_files=True: Saves HTMLs in 'data/' forever.
    keep_files=False: Deletes HTMLs after processing (Clean).
    """
    
    # 1. Setup
    DATA_FOLDER = "data/temporary"
    DB_PATH = vector_store_path
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    company = Company(ticker)
    filings = company.get_filings(form="10-Q")

    if not filings:
        print(f"No SEC filings found for ticker '{ticker}'. It may not be a US-listed company.")
        return None

    filings = filings.latest(8)
    print(f"Found {len(filings)} filings for {ticker}. Processing...")
    all_chunks = [] 

    # 2. Loop through filings
    for filing in tqdm(filings, desc=f"Processing {ticker} Reports", unit="filing"):
        
        # --- FIX: Define ONE consistent path ---
        # We save cleanly as: data/AAPL_2024-01-01.html
        clean_filename = f"{ticker}_{filing.filing_date}.html"
        full_file_path = os.path.join(DATA_FOLDER, clean_filename)
        
        try:
            # A. Get HTML
            html_content = filing.html()
            if not html_content:
                print(f"Skipping {filing.date}: No HTML.")
                continue

            # B. Write to the DATA folder
            with open(full_file_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # C. Load from the DATA folder
            loader = UnstructuredHTMLLoader(full_file_path, mode="elements")
            docs = loader.load()

            # D. Add Metadata
            for doc in docs:
                doc.metadata["ticker"] = ticker
                doc.metadata["date"] = filing.filing_date
                # Parse the date to extract quarter and year
                date_obj = datetime.strptime(str(filing.filing_date), "%Y-%m-%d")
                doc.metadata["year"] = date_obj.year
                doc.metadata["quarter"] = f"Q{(date_obj.month - 1) // 3 + 1}"  # Q1, Q2, Q3, Q4
                doc.metadata["source"] = full_file_path # Point to real file

            # E. Split
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            file_chunks = text_splitter.split_documents(docs)

            # F. Filter short chunks (removes table fragments, headers)
            original_count = len(file_chunks)
            file_chunks = [chunk for chunk in file_chunks if len(chunk.page_content) >= 100]
            short_filtered = original_count - len(file_chunks)

            # G. Filter boilerplate chunks (RAG-004)
            pre_boilerplate = len(file_chunks)
            file_chunks = [chunk for chunk in file_chunks if not is_boilerplate(chunk.page_content)]
            boilerplate_filtered = pre_boilerplate - len(file_chunks)

            if short_filtered > 0 or boilerplate_filtered > 0:
                tqdm.write(f"  Filtered: {short_filtered} short, {boilerplate_filtered} boilerplate")

            all_chunks.extend(file_chunks)
            
            # (Note: We do NOT delete here anymore, we let 'finally' handle it)

        except Exception as e:
            print(f"Error processing {clean_filename}: {e}")

        finally:
            # Clean up temp file if not keeping
            if not keep_files and os.path.exists(full_file_path):
                os.remove(full_file_path)

    # 3. Save to Vector Store (once, after all filings processed)
    if not all_chunks:
        print("No chunks were generated.")
        return

    if os.path.exists(vector_store_path):
        print("Appending to existing Vector Store...")
        vector_store = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        vector_store.add_documents(all_chunks)
    else:
        print("Creating NEW Vector Store...")
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        vector_store = FAISS.from_documents(all_chunks, embedding_model)

    vector_store.save_local(vector_store_path)
    print(f"Success! {len(all_chunks)} chunks saved for {ticker}.")

    # 4. Update ticker_quarters.json with this ticker's quarters
    quarters_set = set()
    for chunk in all_chunks:
        q = chunk.metadata.get("quarter")
        y = chunk.metadata.get("year")
        if q and y:
            quarters_set.add((y, q))  # (2025, "Q3")

    # Sort by year desc, then quarter desc (e.g., Q4 > Q3 > Q2 > Q1)
    sorted_quarters = sorted(quarters_set, key=lambda x: (x[0], int(x[1][1])), reverse=True)

    # Load existing JSON or create new
    if ticker_quarters_path.exists():
        with open(ticker_quarters_path, "r") as f:
            quarters_data = json.load(f)
    else:
        quarters_data = {}
        os.makedirs(os.path.dirname(ticker_quarters_path), exist_ok=True)

    # Update with this ticker's quarters
    quarters_data[ticker] = sorted_quarters

    # Save JSON
    with open(ticker_quarters_path, "w") as f:
        json.dump(quarters_data, f, indent=2)

    print(f"Updated ticker_quarters.json with {len(sorted_quarters)} quarters for {ticker}.")

    # 5. Save chunks to JSON for BM25 keyword search
    chunks_for_bm25 = []
    for chunk in all_chunks:
        # Convert metadata to JSON-serializable format (date objects -> strings)
        metadata = dict(chunk.metadata)
        if "date" in metadata:
            metadata["date"] = str(metadata["date"])
        chunks_for_bm25.append({
            "content": chunk.page_content,
            "metadata": metadata
        })

    # Load existing or create new
    if bm25_chunks_path.exists():
        with open(bm25_chunks_path, "r") as f:
            existing = json.load(f)
        # Filter out old chunks for this ticker (in case of re-ingestion)
        existing = [c for c in existing if c["metadata"]["ticker"] != ticker]
        existing.extend(chunks_for_bm25)
        chunks_for_bm25 = existing
    else:
        os.makedirs(os.path.dirname(bm25_chunks_path), exist_ok=True)

    with open(bm25_chunks_path, "w") as f:
        json.dump(chunks_for_bm25, f)

    print(f"Saved {len(all_chunks)} chunks to bm25_json for {ticker}.")

    return len(all_chunks)