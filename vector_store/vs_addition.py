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
from config import SEC_IDENTITY, EMBEDDING_MODEL, vector_store_path

set_identity(SEC_IDENTITY)


# TODO Add a catch to TypeError: 'NoneType' has no len()  AKA it does not exist

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
    filings = company.get_filings(form="10-Q").latest(8)
    
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
                doc.metadata["source"] = full_file_path # Point to real file

            # E. Split
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            file_chunks = text_splitter.split_documents(docs)
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