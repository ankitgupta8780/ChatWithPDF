import json
import uuid
import hashlib
import os
import io
import sqlite3
from datetime import datetime, timedelta
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


@st.cache_data
def process_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))

    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunk_size = 500
    chunks = []

    words = full_text.split()

    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current_chunk))

            current_chunk = current_chunk[-10:]
            current_length = sum(len(w) + 1 for w in current_chunk)

        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if not chunks:
        return [], None

    model = load_model()
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index

def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            messages TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()

def save_chat(chat_id, title, messages):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO chats
        (id, title, messages, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        chat_id,
        title,
        json.dumps(messages),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

def get_recent_chats():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title, messages, created_at
        FROM chats
        WHERE created_at >= ?
        ORDER BY created_at DESC
    """, (
        (datetime.now() - timedelta(days=7)).isoformat(),
    ))

    chats = cursor.fetchall()
    conn.close()

    return chats

def delete_expired_chats():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM chats
        WHERE created_at < ?
    """, (
        (datetime.now() - timedelta(days=7)).isoformat(),
    ))

    conn.commit()
    conn.close()


delete_expired_chats()

def delete_chat(chat_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM chats WHERE id = ?",
        (chat_id,)
    )

    conn.commit()
    conn.close()

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0B0F19;
    color: #E5E7EB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #273244;
}

/* Main content width */
.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3 {
    color: #F9FAFB !important;
}

/* Buttons */
.stButton > button {
    background-color: #1F2937;
    color: #E5E7EB;
    border: 1px solid #374151;
    border-radius: 10px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #374151;
    color: white;
    border-color: #818CF8;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 12px;
}

[data-testid="stChatInput"] textarea {
    color: #E5E7EB !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9CA3AF !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 14px;
    padding: 12px;
}
/* Sidebar spacing */
[data-testid="stSidebar"] {
    padding-top: 1.5rem;
}

/* Sidebar heading */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 1.2rem;
    font-weight: 650;
    letter-spacing: 0.3px;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    min-height: 42px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 500;
}
/* Main header */
.main-header {
    margin-bottom: 30px;
}

.main-header h1 {
    font-size: 2.4rem;
    font-weight: 750;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.main-header p {
    color: #9CA3AF;
    font-size: 1rem;
    margin-top: 0;
}
/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #111827;
    border: 1px solid #273244;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Chat message text */
[data-testid="stChatMessage"] p {
    color: #E5E7EB;
    line-height: 1.6;
}
/* Success and info messages */
[data-testid="stAlert"],
[data-testid="stAlert"] > div,
[data-testid="stAlert"] [role="alert"] {
    background-color: #111827 !important;
    background: #111827 !important;
    border: 1px solid #273244 !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] * {
    color: #E5E7EB !important;
}
/* Welcome section */
.welcome-section {
    text-align: center;
    padding: 70px 20px;
    margin-top: 30px;
}

.welcome-section h2 {
    font-size: 1.7rem;
    font-weight: 650;
    color: #F9FAFB;
    margin-bottom: 12px;
}

.welcome-section p {
    color: #D1D5DB;
    font-size: 1rem;
}

.welcome-section .welcome-hint {
    color: #9CA3AF;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)
    
with st.sidebar:
    st.subheader("🕘 Recent Chats")

    recent_chats = get_recent_chats()

    if not recent_chats:
        st.write("No recent chats.")

    for chat in recent_chats:
        chat_id, title, messages, created_at = chat

        col1, col2 = st.columns([4, 1])

        with col1:
            if st.button(title, key=f"open_{chat_id}"):
                 st.session_state.chat_id = chat_id
                 st.session_state.chat_title = title
                 st.session_state.messages = json.loads(messages)
                 st.session_state.opened_chat = True
                 st.rerun()

        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()



if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "chat_title" not in st.session_state:
    st.session_state.chat_title = None


api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("GROQ_API_KEY is not configured.")
        st.stop()

client = Groq(api_key=api_key)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


st.markdown("""
<div class="main-header">
    <h1>📄 Chat with PDF</h1>
    <p>Upload your document and get instant answers powered by AI.</p>
</div>
""", unsafe_allow_html=True)
st.subheader("Upload your PDF")
st.caption("Add a document to start asking questions.")

uploaded_file = st.file_uploader(
    "Browse files",
    type="pdf"
)

if uploaded_file is not None:

    pdf_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.get("pdf_hash") != pdf_hash:
        st.session_state.pdf_hash = pdf_hash
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.chat_title = uploaded_file.name
        st.session_state.messages = []

    with st.spinner("Processing PDF..."):
        chunks, index = process_pdf(uploaded_file.getvalue())

    if not chunks:
        st.error("No readable text found in this PDF.")
        st.stop()

    st.success("PDF is ready to chat!")

    if chunks and index is not None:


        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input stays at the bottom
        question = st.chat_input("Ask something about your PDF...")

        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
           

            with st.chat_message("user"):
                st.write(question)

           
            model = load_model()
            question_embedding = model.encode([question])
            distances, indices = index.search(
                question_embedding, k=4
            )

            context = "\n\n".join(
                chunks[idx]
                for idx in indices[0]
                if idx != -1
            )

            try:
                with st.spinner("AI is thinking..."):
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer using only the provided PDF context. "
                                    "If the answer is not present, say you "
                                    "couldn't find it in the PDF."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"PDF context:\n{context}\n\n"
                                    f"Question: {question}"
                                )
                            }
                        ]
                    )

                answer = response.choices[0].message.content

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                save_chat(
                    st.session_state.chat_id,
                    st.session_state.chat_title,
                    st.session_state.messages
                )

                st.rerun()

            except Exception as e:
                st.error(f"Could not generate an answer: {e}")

else:
    if st.session_state.get("opened_chat", False):
        st.subheader(st.session_state.chat_title)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    else:
        st.markdown("""
<div class="welcome-section">
    <h2>Welcome to Chat with PDF 👋</h2>
    <p>Upload a PDF document and ask questions about its content.</p>
    <p class="welcome-hint">Your AI assistant will find answers directly from your document.</p>
</div>
""", unsafe_allow_html=True)







