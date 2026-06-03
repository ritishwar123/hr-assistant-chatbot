import os
import re
from typing import List, Dict, Any, Tuple
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Cache representation of the embedding model to optimize startup and script reruns
_embedding_model_cache = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer model to prevent reloading
    on every Streamlit script execution rerun.
    """
    global _embedding_model_cache
    if _embedding_model_cache is None:
        # Load the sentence-transformer model (384-dimensional vectors)
        _embedding_model_cache = SentenceTransformer(model_name)
    return _embedding_model_cache


from chromadb import EmbeddingFunction

class HFChromaEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding class for ChromaDB to use the cached local sentence-transformer.
    ChromaDB expects a callable that takes a list of texts and returns a list of embeddings.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = get_embedding_model(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Generate embeddings and convert them to standard Python float lists
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, input: List[str]) -> List[List[float]]:
        # Required by ChromaDB's internal query validation workflow
        return self.__call__(input)


def extract_text_from_pdfs(pdf_files: List[Any]) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from uploaded PDF file objects.
    
    Args:
        pdf_files: List of file-like objects (e.g., BytesIO from st.file_uploader)
        
    Returns:
        A list of dictionaries containing text, document name, and page number.
    """
    pages_data = []
    
    for pdf_file in pdf_files:
        # Streamlit uploads are BytesIO objects.
        # We pass them directly to PyPDF's PdfReader.
        reader = pypdf.PdfReader(pdf_file)
        raw_name = getattr(pdf_file, "name", "unknown_document.pdf")
        # Extract base name to keep citations clean and short
        doc_name = os.path.basename(raw_name) if isinstance(raw_name, str) else "unknown_document.pdf"
        
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                # Clean up any potential double spacing or weird characters
                cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
                pages_data.append({
                    "text": cleaned_text,
                    "source": doc_name,
                    "page": page_idx + 1
                })
                
    return pages_data


def chunk_document_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    Splits a continuous string of text into overlapping chunks.
    This implementation attempts to break at word boundaries to keep chunks semantic.
    
    Args:
        text: The text content of a single page or document.
        chunk_size: Maximum character length of a single chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        prev_start = start
        # Calculate target end
        end = min(start + chunk_size, text_len)
        
        # If we are not at the end of the document, try to split at a space
        if end < text_len:
            # Look backwards from the target end for a space to split cleanly
            # Limit the search range to avoid creating too-small chunks
            search_start = max(start, end - 100)
            last_space = text.rfind(' ', search_start, end)
            if last_space != -1:
                end = last_space
                
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # If we have reached the end of the text, stop chunking
        if end == text_len:
            break
            
        # Move start point back by the overlap amount
        start = end - chunk_overlap
        
        # Ensure we always make forward progress (avoid infinite loops/stagnation)
        if start <= prev_start:
            start = prev_start + 1
            
    return chunks


def prepare_chunks_for_indexing(pages_data: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Processes extracted pages and chunks their text while preserving metadata
    about the source document and page number.
    """
    chunked_documents = []
    
    for page_data in pages_data:
        text = page_data["text"]
        source = page_data["source"]
        page_num = page_data["page"]
        
        chunks = chunk_document_text(text, chunk_size, chunk_overlap)
        
        for idx, chunk in enumerate(chunks):
            chunked_documents.append({
                "text": chunk,
                "metadata": {
                    "source": source,
                    "page": page_num,
                    "chunk_index": idx
                }
            })
            
    return chunked_documents


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Returns a persistent ChromaDB client pointing to a local workspace folder.
    """
    persist_dir = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "scratch", "hr_assistant_chatbot", "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def index_documents_to_chroma(collection_name: str, chunks: List[Dict[str, Any]]) -> Any:
    """
    Creates or updates a ChromaDB collection and indexes the document chunks with embeddings.
    """
    client = get_chroma_client()
    
    # If the collection already exists, we delete it to ensure a fresh session upload.
    try:
        client.delete_collection(collection_name)
    except Exception:
        # Collection didn't exist, which is fine
        pass
        
    # Get our custom HuggingFace embedding function
    embedding_fn = HFChromaEmbeddingFunction()
    
    # Create a fresh collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity for distance calculations
    )
    
    # Prepare batch uploads
    ids = []
    documents = []
    metadatas = []
    
    for idx, item in enumerate(chunks):
        ids.append(f"chunk_{idx}")
        documents.append(item["text"])
        metadatas.append(item["metadata"])
        
    # Chroma DB allows bulk addition
    if documents:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
    return collection


def query_rag_engine(collection_name: str, query: str, gemini_api_key: str, model_name: str = "gemini-2.5-flash", k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Performs retrieval on ChromaDB, builds a grounded prompt, queries the Gemini LLM,
    and returns the grounded text response along with retrieved sources.
    """
    client = get_chroma_client()
    embedding_fn = HFChromaEmbeddingFunction()
    
    try:
        collection = client.get_collection(name=collection_name, embedding_function=embedding_fn)
    except Exception:
        return "No documents have been indexed yet. Please upload documents in the sidebar.", []
        
    # 1. Retrieve the top K matching documents
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    
    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return "I could not find the information in the provided HR documents.", []
        
    # 2. Extract retrieved text blocks and source metadata
    retrieved_docs = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    retrieved_distances = results["distances"][0]
    
    context_blocks = []
    sources = []
    
    for doc, meta, dist in zip(retrieved_docs, retrieved_metadatas, retrieved_distances):
        # We can set a soft threshold on distance (e.g., for cosine, distance > 0.8 usually indicates very low relevance)
        # Cosine distance in Chroma is 1 - CosineSimilarity. So distance of 0.0 is exact match, 1.0 is orthogonal.
        # We include the chunk text in the prompt context
        context_blocks.append(f"Source: {meta['source']}, Page: {meta['page']}\nContent: {doc}")
        sources.append({
            "source": meta["source"],
            "page": meta["page"],
            "distance": dist
        })
        
    # Format context for prompt
    context_text = "\n\n===\n\n".join(context_blocks)
    
    # 3. Formulate the strictly grounded system prompt for Gemini
    system_instruction = (
        "You are an expert HR Assistant. Your primary responsibility is to answer user questions using ONLY "
        "the provided HR document excerpts. You must follow these rules strictly:\n\n"
        "1. Base your answer ONLY on the context snippets provided below. Do not use external knowledge or assume facts outside this context.\n"
        "2. If the answer cannot be found or directly inferred from the provided context, you MUST state exactly: "
        "'I could not find the information in the provided HR documents.' and nothing else. Do not add general advice or speculative answers.\n"
        "3. Provide inline citations to the documents if possible, referencing the document name and page number.\n"
        "4. Keep your answer factual, professional, and clear. Do not mention that you are an AI model or refer to 'the provided snippets' in your final response to the employee; present the answer naturally but strictly grounded."
    )
    
    prompt = f"""HR Document Context:
{context_text}

Question: {query}

Answer:"""

    # 4. Invoke Gemini API
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0, # Zero temperature is critical for deterministic, factual extraction in RAG
                max_output_tokens=1024
            )
        )
        
        answer = response.text.strip()
        
        # Double check model compliance on negative answers.
        # Sometimes models add a small buffer text. We want to align with the prompt requirements.
        if "could not find the information" in answer.lower():
            # Standardize output
            answer = "I could not find the information in the provided HR documents."
            
        return answer, sources
        
    except Exception as e:
        return f"An error occurred while connecting to the Gemini LLM API: {str(e)}", []
