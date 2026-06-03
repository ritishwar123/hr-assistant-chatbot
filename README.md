# HR Assistant Chatbot - Enterprise RAG Architecture

This repository contains a production-grade, Retrieval-Augmented Generation (RAG) chatbot designed for corporate HR inquiries. It utilizes **Streamlit** for the frontend UI, **ChromaDB** for vector storage, **HuggingFace SentenceTransformers** for local embedding computation, and **Gemini** for strictly grounded LLM responses.

---

## 1. System Architecture & Lifecycle Flow

The application is split into a modular RAG core backend (`rag_engine.py`) and a stateless frontend server (`app.py`). 

```
                                  [ Upload Workflow ]
                             
   ┌─────────────┐        ┌─────────────┐        ┌────────────────┐        ┌─────────────┐
   │ PDF Uploads │ ─────> │ PyPDF text  │ ─────> │ Recursive Text │ ─────> │ Embedding   │
   │ (Streamlit) │        │ extraction  │        │ splitting      │        │ Gen (Local) │
   └─────────────┘        └─────────────┘        └────────────────┘        └─────────────┘
                                                                                  │
                                                                                  ▼
                                                                           ┌─────────────┐
                                                                           │  ChromaDB   │
                                                                           │ (Persistent)│
                                                                           └─────────────┘
                                                                                  
                                   [ Query Workflow ]
                                   
   ┌─────────────┐        ┌─────────────┐        ┌────────────────┐        ┌─────────────┐
   │ User Query  │ ─────> │  Vector DB  │ ─────> │ Context &      │ ─────> │ Gemini LLM  │
   │ (Chat Input)│        │ Cosine Query│        │ Grounded Prompt│        │ (Zero Temp) │
   └─────────────┘        └─────────────┘        └────────────────┘        └─────────────┘
                                                          ▲                       │
                                                          │                       ▼
                                                   ┌──────────────┐        ┌─────────────┐
                                                   │ Citations /  │        │ Grounded HR │
                                                   │ Sources List │ <───── │ Response    │
                                                   └──────────────┘        └─────────────┘
```

### Process Lifecycle:
1. **Document Loading**: Multiple PDFs are read into memory. Since Streamlit uploads are byte streams, `pypdf.PdfReader` processes them directly from memory to avoid disk write overhead.
2. **Text Parsing & Metadata Injection**: Text is extracted page-by-page. Metadata records tracking the source file name and page numbers are pinned to every text block.
3. **Recursive Semantic Chunking**: The page content is split using an overlap sliding window. We check character positions and backtrack to space boundaries to prevent cutting off words or sentences.
4. **Vector Database Ingestion**: An in-memory/persistent ChromaDB collection is instantiated. Chunks are embedded on-the-fly and stored in a indexed space.
5. **Semantic Retrieval**: User questions are converted to embeddings. The database returns the top $K$ ($K=5$) nearest neighbors using Cosine similarity.
6. **Prompt Synthesis & LLM Inference**: The retrieved chunks are formatted into an isolated block inside the system prompt. The model processes the query at `temperature = 0.0` to force factual extraction.

---

## 2. Technical Deep-Dive & Code Rationale

### A. Thread-Safe Session Isolation in Multi-Tenant Environments
*   **The Problem**: A single Streamlit application container runs as a shared server process serving multiple users concurrently. Using a static ChromaDB collection name would lead to cross-user data leaks, where Employee A retrieves HR documents uploaded by Employee B.
*   **The Solution**: We generate a unique `session_id` stored in `st.session_state` upon connection initialization:
    ```python
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"col_{uuid.uuid4().hex[:20]}"
    ```
    This UUID is used as the ChromaDB collection name. When the session terminates or is cleared, the collection is deleted, ensuring total security and session isolation.

### B. Embedding Caching & Optimization
*   **The Problem**: Instantiating a `SentenceTransformer("all-MiniLM-L6-v2")` model on every Streamlit script execution rerun (which triggers on every click or text input) compiles PyTorch weights repeatedly. This causes high CPU load and input lag.
*   **The Solution**: We cache the model at the module level:
    ```python
    _embedding_model_cache = None

    def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        global _embedding_model_cache
        if _embedding_model_cache is None:
            _embedding_model_cache = SentenceTransformer(model_name)
        return _embedding_model_cache
    ```
    This acts as a singleton, ensuring the weights are loaded into memory exactly once.

### C. Chunking Edge-Case Protection
*   **The Problem**: Standard naive chunking splits text on fixed character lengths. This breaks words, disrupts grammatical structures, and can result in infinite loops if the text length is shorter than the target chunk overlap.
*   **The Solution**: Our custom splitting logic tracks boundary spaces (`text.rfind(' ')`) to ensure clean word breaks. It also explicitly tracks the previous iteration's start pointer to guarantee forward progress:
    ```python
    if end == text_len:
        break
    start = end - chunk_overlap
    if start <= prev_start:
        start = prev_start + 1
    ```
    This safeguards the loop from hanging on small pages.

### D. Strict Hallucination Grounding
To prevent the model from leaking external training knowledge or speculating when policies are missing:
1.  **System Instruction Constraints**: The model is configured with a strict system prompt stating: *'If the answer cannot be found or directly inferred from the provided context, you MUST state exactly: "I could not find the information in the provided HR documents." and nothing else.'*
2.  **Zero Temperature (`temperature = 0.0`)**: This disables top-k/top-p sampling, forcing the model to take the path of maximum likelihood, resulting in deterministic factual extraction.
3.  **Strict post-processing validation**: A backend check checks for negative indicators in the output to clean and normalize negative responses.

---

## 3. Local Installation & Development

### 1. Prerequisite Checks
Make sure you have Python 3.10 to 3.12 installed.

### 2. Install Project Dependencies
Run the installation in your terminal:
```bash
pip install -r requirements.txt
```

### 3. Setup Test Documents
Create sample test PDFs using our generator:
```bash
python scratch/generate_test_pdfs.py
```
This populates the `sample_docs/` folder with:
*   `acme_leave_policy.pdf`
*   `acme_employee_handbook.pdf`

### 4. Run the Pipeline Validation Test
If you want to run the vector search pipeline locally in your terminal:
```bash
# Optional: set API key in environment to test LLM step
# set GEMINI_API_KEY="your_api_key_here"
python scratch/test_rag.py
```

### 5. Launch the Streamlit Chatbot
Run the Streamlit application:
```bash
streamlit run app.py
```

---

## 4. Production Deployment to Streamlit Cloud

To deploy this application to **Streamlit Cloud**:

1.  **Repository Setup**: Push the files to a public or private GitHub repository.
2.  **Streamlit Cloud Registration**: Link your GitHub account to [share.streamlit.io](https://share.streamlit.io/).
3.  **Configure API Secrets**:
    *   In the Streamlit Cloud dashboard, open your app settings.
    *   Navigate to the **Secrets** section.
    *   Input your Gemini API Key in the following format:
        ```toml
        GEMINI_API_KEY = "AIzaSy..."
        ```
    *   The application will automatically pick up this secret using `st.secrets["GEMINI_API_KEY"]` on startup, allowing employees to start chatting immediately without needing to enter keys manually!
