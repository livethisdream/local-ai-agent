import streamlit as st
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
import uuid
import glob

# --- CONFIGURATION ---
DB_PATH = "./rag_db"
COLLECTION_NAME = "rag_knowledge"
MODEL_NAME = "deepseek-coder:6.7b"  # The Base Model
SESSIONS_DIR = "rag_sessions"  # Where to store chat files

st.set_page_config(page_title="Pure RAG Explorer", layout="wide")

# Ensure sessions directory exists
os.makedirs(SESSIONS_DIR, exist_ok=True)


# --- 1. RESOURCES ---
@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, embedder


try:
    collection, embedder = load_resources()
except Exception as e:
    st.error(f"⚠️ Database not found at {DB_PATH}. Did you run 'ingest_data.py'?")
    st.stop()


# --- 2. SESSION MANAGEMENT ---
def get_all_sessions():
    """Returns a dict of {session_id: {'file': path, 'title': title}}"""
    files = glob.glob(f"{SESSIONS_DIR}/*.json")
    sessions = {}
    files.sort(key=os.path.getmtime, reverse=True)  # Newest first

    for f in files:
        try:
            with open(f, "r") as file:
                data = json.load(file)
                title = "New Chat"
                if len(data) > 0:
                    # Find first user msg for title
                    for msg in data:
                        if msg["role"] == "user":
                            title = msg["content"][:25] + "..."
                            break
                session_id = os.path.basename(f).replace(".json", "")
                sessions[session_id] = {"file": f, "title": title}
        except:
            pass
    return sessions


def load_session(session_id):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_session(session_id, messages):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(path, "w") as f:
        json.dump(messages, f)


def create_new_session():
    new_id = str(uuid.uuid4())
    save_session(new_id, [])
    return new_id


# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Chat Sessions")

    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_new_session()
        st.session_state["active_session"] = new_id
        st.rerun()

    st.divider()

    # RAG Controls
    st.subheader("⚙️ RAG Settings")
    use_rag = st.checkbox("Enable Search", value=True)
    k_retrieval = st.slider("Documents to read", 1, 5, 3)

    st.divider()

    # Session List
    sessions = get_all_sessions()
    session_ids = list(sessions.keys())
    session_titles = [sessions[k]["title"] for k in session_ids]

    # Init active session
    if "active_session" not in st.session_state:
        if session_ids:
            st.session_state["active_session"] = session_ids[0]
        else:
            st.session_state["active_session"] = create_new_session()
            st.rerun()

    # Sync radio button with state
    current_index = 0
    if st.session_state["active_session"] in session_ids:
        current_index = session_ids.index(st.session_state["active_session"])

    selected_title = st.radio(
        "History",
        session_titles,
        index=current_index,
        label_visibility="collapsed"
    )

    # Update state based on selection
    for sid, info in sessions.items():
        if info["title"] == selected_title:
            st.session_state["active_session"] = sid
            break

# --- 4. MAIN CHAT ---
active_id = st.session_state["active_session"]
messages = load_session(active_id)

st.title(f"💬 {sessions[active_id]['title']}")

# Display History
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Loop
if user_input := st.chat_input("Ask about the codebase..."):
    # 1. User Message
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    save_session(active_id, messages)

    # 2. RAG Retrieval
    context_text = ""
    sources = set()

    if use_rag:
        with st.spinner("Searching docs..."):
            query_vec = embedder.encode([user_input]).tolist()
            results = collection.query(query_embeddings=query_vec, n_results=k_retrieval)

            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    src = os.path.basename(results['metadatas'][0][i]['source'])
                    context_text += f"-- FILE: {src} --\n{doc}\n\n"
                    sources.add(src)

    # 3. Generation
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_resp = ""

        # Prompt Logic
        if context_text:
            sys_msg = (
                "You are an expert developer. "
                "Use the Context below to answer. "
                "If the context is irrelevant, ignore it and use your own knowledge."
            )
            final_prompt = f"{sys_msg}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{user_input}"
        else:
            final_prompt = user_input

        # Stream Response
        stream = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': final_prompt}], stream=True)

        for chunk in stream:
            content = chunk['message']['content']
            full_resp += content
            placeholder.markdown(full_resp + "▌")

        # Append Citations
        if sources:
            full_resp += "\n\n**📚 Sources:** " + ", ".join([f"`{s}`" for s in sources])
            placeholder.markdown(full_resp)

    # 4. Save Assistant Message
    messages.append({"role": "assistant", "content": full_resp})
    save_session(active_id, messages)

    # Refresh title if it's the first message
    if len(messages) == 2:
        st.rerun()