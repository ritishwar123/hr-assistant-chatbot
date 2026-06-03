import os
import uuid
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables (if any)
load_dotenv()

# Import our custom RAG engine
from rag_engine import (
    extract_text_from_pdfs,
    prepare_chunks_for_indexing,
    index_documents_to_chroma,
    query_rag_engine
)

# 1. Set Page Configuration with modern title and icon
st.set_page_config(
    page_title="HR Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Premium Custom CSS for Rich Aesthetics & Glassmorphism
st.markdown("""
<style>
    /* Import modern Outfit font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply font family globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Set custom dark-themed radial background for the app */
    .stApp {
        background: radial-gradient(circle at 80% 20%, #151622 0%, #0c0d12 100%);
        color: #e2e8f0;
    }
    
    /* Styled Sidebar with Glassmorphic feel */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 16, 25, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Gradient text title */
    .gradient-title {
        background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05em;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Custom Chat Interface Styling */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        padding: 1rem 0;
        max-width: 850px;
        margin: 0 auto;
    }
    
    /* User Message Bubble */
    .chat-bubble-user {
        align-self: flex-end;
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: #ffffff;
        padding: 0.9rem 1.3rem;
        border-radius: 18px 18px 2px 18px;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2);
        animation: slideInRight 0.3s ease-out;
        font-size: 1.05rem;
        line-height: 1.5;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Assistant Message Bubble (Glassmorphic) */
    .chat-bubble-assistant {
        align-self: flex-start;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #f1f5f9;
        padding: 1.1rem 1.4rem;
        border-radius: 18px 18px 18px 2px;
        max-width: 75%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
        animation: slideInLeft 0.3s ease-out;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    /* Metainfo style */
    .message-meta {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .user-meta {
        justify-content: flex-end;
    }
    
    /* Key animations */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateY(10px) translateX(10px); }
        to { opacity: 1; transform: translateY(0) translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateY(10px) translateX(-10px); }
        to { opacity: 1; transform: translateY(0) translateX(0); }
    }
    
    /* Source Pill styling */
    .source-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: #818cf8;
        margin-top: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .source-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.4rem;
    }
    
    .source-pill {
        background-color: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #c7d2fe;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
        transition: all 0.2s ease;
    }
    
    .source-pill:hover {
        background-color: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        cursor: default;
    }
    
    /* Sidebar info boxes */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Styling Streamlit UI buttons to match styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 15px rgba(79, 70, 229, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# 3. Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
    
if "session_id" not in st.session_state:
    # Generate unique ID for this session to isolate the vector store collection
    st.session_state.session_id = f"col_{uuid.uuid4().hex[:20]}"


# 4. Resolve Gemini API Key (Secrets -> Environment -> Sidebar input)

gemini_api_key = ""

# First try Streamlit secrets
try:
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    gemini_api_key = ""

# Then try environment variable
if not gemini_api_key:
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")


# 5. Sidebar Layout
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/000000/manager.png", width=80)
    st.markdown("### System Settings")
    
    # API Key Input if not already resolved
    if not gemini_api_key:
        user_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Get your key from Google AI Studio. It will only be stored for this browser session."
        )
        if user_key:
            gemini_api_key = user_key
            
    if not gemini_api_key:
        st.warning("⚠️ Please provide a Gemini API Key to activate the chat assistant.")
    else:
        st.success("🔑 Gemini API Key configured.")
        
    st.markdown("---")
    st.markdown("### Knowledge Base Upload")
    
    uploaded_files = st.file_uploader(
        "Upload HR Policy PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload company handbooks, leave details, health insurance details, etc."
    )
    
    process_button = st.button("Index Documents")
    
    if process_button and uploaded_files:
        if not gemini_api_key:
            st.error("Please configure the Gemini API Key first.")
        else:
            with st.spinner("Extracting text from PDFs..."):
                pages_data = extract_text_from_pdfs(uploaded_files)
                
            if pages_data:
                with st.spinner("Chunking text passages..."):
                    # Split pages into chunk sizes of 800 chars with 150 overlap
                    chunks = prepare_chunks_for_indexing(pages_data, chunk_size=800, chunk_overlap=150)
                    
                with st.spinner("Generating embeddings & Indexing to ChromaDB..."):
                    # Index chunks to session-specific collection
                    index_documents_to_chroma(st.session_state.session_id, chunks)
                    
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.success(f"Successfully indexed {len(chunks)} chunks from {len(uploaded_files)} document(s)!")
            else:
                st.error("Failed to extract text from the uploaded PDFs. Please make sure they are not scanned/image-only PDFs.")
                
    st.markdown("---")
    
    # Display details of currently indexed documents
    if st.session_state.indexed_files:
        st.markdown("#### Indexed Knowledge Source:")
        for filename in st.session_state.indexed_files:
            st.markdown(f"📄 **{filename}**")
            
        # Reset button to clear vector DB and state
        if st.button("Clear Indexed Files"):
            st.session_state.indexed_files = []
            st.session_state.chat_history = []
            # Generate new session ID to abandon old collection
            st.session_state.session_id = f"col_{uuid.uuid4().hex[:20]}"
            st.rerun()
    else:
        st.info("No documents indexed. Upload PDFs and click 'Index Documents' to build the HR knowledge base.")
        
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.8rem; color: #64748b; text-align: center;'>"
        "Built with Antigravity & Streamlit<br>Powered by Gemini & HuggingFace</div>", 
        unsafe_allow_html=True
    )


# 6. Main Chat Area
st.markdown("<div class='gradient-title'>HR Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Corporate RAG Bot - Accurate, Grounded, Secure HR Guidance</div>", unsafe_allow_html=True)

# 7. Render Chat History in custom styled containers
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        role = message["role"]
        text = message["text"]
        
        if role == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{text}'
                f'<div class="message-meta user-meta">👤 You</div></div>',
                unsafe_allow_html=True
            )
        else:
            sources_html = ""
            if "sources" in message and message["sources"]:
                # Build beautiful pill markup for the citations
                pills = []
                # Remove duplicate source-page mappings for clean UX
                seen_sources = set()
                for src in message["sources"]:
                    src_key = f"{src['source']}_p{src['page']}"
                    if src_key not in seen_sources:
                        seen_sources.add(src_key)
                        pills.append(
                            f'<span class="source-pill">📄 {src["source"]} (Page {src["page"]})</span>'
                        )
                if pills:
                    sources_html = (
                        f'<div class="source-header">Sources Verified:</div>'
                        f'<div class="source-container">{" ".join(pills)}</div>'
                    )
            
            st.markdown(
                f'<div class="chat-bubble-assistant">{text}{sources_html}'
                f'<div class="message-meta">🤖 HR Assistant</div></div>',
                unsafe_allow_html=True
            )
            
    st.markdown("</div>", unsafe_allow_html=True)
else:
    # Premium Welcome Display
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 2.5rem; max-width: 850px; margin: 2rem auto; text-align: center;">
        <h3 style="margin-top: 0;">Welcome to your Internal HR Assistant</h3>
        <p style="color: #94a3b8; font-size: 1.05rem; line-height: 1.6; max-width: 600px; margin: 0 auto 1.5rem auto;">
            This system utilizes Retrieval-Augmented Generation (RAG) to scan company documentation and provide 100% grounded answers. 
            No hallucinations, no external data leakage.
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <div style="background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 10px; flex: 1; min-width: 200px;">
                <h4 style="margin: 0 0 0.5rem 0; color: #818cf8 !important;">1. Upload Documents</h4>
                <span style="font-size: 0.9rem; color: #94a3b8;">Supply leave policies, company guidelines, or handbook PDFs in the sidebar.</span>
            </div>
            <div style="background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 10px; flex: 1; min-width: 200px;">
                <h4 style="margin: 0 0 0.5rem 0; color: #818cf8 !important;">2. Ask Questions</h4>
                <span style="font-size: 0.9rem; color: #94a3b8;">Inquire about leave rules, health benefits, or remote work schedules.</span>
            </div>
            <div style="background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 10px; flex: 1; min-width: 200px;">
                <h4 style="margin: 0 0 0.5rem 0; color: #818cf8 !important;">3. Grounded Verification</h4>
                <span style="font-size: 0.9rem; color: #94a3b8;">Answers are mapped directly to source document citations.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# 8. Chat Input Form
# Anchor it visually at the bottom
query = st.chat_input("Ask a question about corporate HR policies...")

if query:
    if not st.session_state.indexed_files:
        st.warning("⚠️ Please upload and index HR documents in the sidebar first before asking questions.")
    elif not gemini_api_key:
        st.warning("⚠️ Please provide a Gemini API Key in the sidebar to process your query.")
    else:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "text": query})
        
        # Display immediately
        st.rerun()

# 9. Handle RAG processing if the last message is from the user
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    user_query = st.session_state.chat_history[-1]["text"]
    
    with st.spinner("Consulting HR documents..."):
        answer, sources = query_rag_engine(
            st.session_state.session_id, 
            user_query, 
            gemini_api_key
        )
        
    st.session_state.chat_history.append({
        "role": "assistant",
        "text": answer,
        "sources": sources
    })
    
    # Rerun to update chat screen
    st.rerun()
