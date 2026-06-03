import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path so we can import from our backend engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from rag_engine import (
    extract_text_from_pdfs,
    prepare_chunks_for_indexing,
    index_documents_to_chroma,
    query_rag_engine
)

def run_test():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(project_dir, "sample_docs")
    
    # 1. Locate PDF files
    print("--- STEP 1: Locating PDF files ---")
    pdf_paths = [
        os.path.join(docs_dir, "acme_leave_policy.pdf"),
        os.path.join(docs_dir, "acme_employee_handbook.pdf")
    ]
    
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"Error: {path} does not exist. Run generate_test_pdfs.py first.")
            return
    print(f"Found {len(pdf_paths)} sample documents.")
    
    # 2. Extract Text
    print("\n--- STEP 2: Extracting text from PDFs ---")
    # Open the PDFs in binary mode and simulate uploaded file objects
    file_objs = []
    for path in pdf_paths:
        f = open(path, "rb")
        file_objs.append(f)
        
    try:
        pages_data = extract_text_from_pdfs(file_objs)
        print(f"Extracted {len(pages_data)} pages of text.")
    finally:
        for f in file_objs:
            f.close()
            
    # 3. Chunk Text
    print("\n--- STEP 3: Splitting text into chunks ---")
    chunks = prepare_chunks_for_indexing(pages_data, chunk_size=800, chunk_overlap=150)
    print(f"Generated {len(chunks)} chunks.")
    print(f"Sample Chunk 0:\nSource: {chunks[0]['metadata']['source']} (Page {chunks[0]['metadata']['page']})")
    print(f"Content: {chunks[0]['text'][:200]}...")
    
    # 4. Index Chunks
    print("\n--- STEP 4: Creating ChromaDB vector index ---")
    test_collection = "test_hr_docs"
    collection = index_documents_to_chroma(test_collection, chunks)
    print(f"ChromaDB collection '{test_collection}' created and loaded.")
    
    # 5. Query Vector Database
    print("\n--- STEP 5: Testing Vector Search (Retrieval) ---")
    test_query = "How many sick leave days do I get and do I need a note?"
    print(f"Query: '{test_query}'")
    
    # Let's perform direct query on the collection to verify search
    # This checks our embedding + search pipeline works offline without Gemini
    results = collection.query(
        query_texts=[test_query],
        n_results=2
    )
    
    print("\nRetrieved Chunks:")
    for idx, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
        print(f"\nResult #{idx+1}:")
        print(f"  Source: {meta['source']} (Page {meta['page']})")
        print(f"  Distance (lower is better): {dist:.4f}")
        print(f"  Text: {doc[:300]}...")
        
    # 6. Optional: Gemini LLM Grounded Query
    print("\n--- STEP 6: Testing LLM generation (RAG) ---")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("Note: GEMINI_API_KEY environment variable not set. Skipping Gemini generation check.")
        print("To test LLM generation, set GEMINI_API_KEY in a .env file in the root directory.")
    else:
        print("GEMINI_API_KEY found. Running Gemini generation...")
        answer, sources = query_rag_engine(test_collection, test_query, gemini_key)
        print("\nGemini Answer:")
        print(answer)
        print("\nVerified sources cited in LLM query:")
        for src in sources:
            print(f"  - {src['source']} (Page {src['page']}) - Distance: {src['distance']:.4f}")
            
        print("\n--- Testing Grounding (Out of boundary query) ---")
        out_of_bound_query = "What is the policy for buying cryptocurrency?"
        print(f"Query: '{out_of_bound_query}'")
        answer_out, _ = query_rag_engine(test_collection, out_of_bound_query, gemini_key)
        print("Gemini Answer:")
        print(answer_out)
        
if __name__ == "__main__":
    run_test()
