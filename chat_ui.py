import streamlit as st
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
import uuid
import glob
from datetime import datetime

# --- CONFIG ---
DB_PATH = "./rag_db"
COLLECTION_NAME = "gnuradio_knowledge"
MODEL_NAME = "gnuradio-coder"
SESSIONS_DIR = "chat_sessions"

st.set_page_config(page_title="GNURadio Agent", layout="wide")

# Ensure sessions directory exists
os.makedirs(SESSIONS_DIR, exist_ok=True)


# --- 1. LOAD AI RESOURCES (Cached) ---
@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, embedder


collection, embedder = load_resources()


# --- 2. SESSION MANAGEMENT ---
def get_all_sessions():
    """Returns a dict of {session_id: {'filename': f, 'title': t}}"""
    files = glob.glob(f"{SESSIONS_DIR}/*.json")
    sessions = {}
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)

    for f in files:
        try:
            with open(f, "r") as file:
                data = json.load(file)
                # Use the first user message as the title, or "Empty Chat"
                title = "New Chat"
                if len(data) > 0:
                    for msg in data:
                        if msg["role"] == "user":
                            title = msg["content"][:30] + "..."  # Truncate
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


# --- 3. SIDEBAR: CHAT LIST ---
with st.sidebar:
    st.title("🗂️ Chats")

    # "New Chat" Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_new_session()
        st.session_state["active_session"] = new_id
        st.rerun()

    st.divider()

    # List Existing Chats
    sessions = get_all_sessions()
    session_ids = list(sessions.keys())
    session_titles = [sessions[k]["title"] for k in session_ids]

    # Initialize active session if none exists
    if "active_session" not in st.session_state:
        if session_ids:
            st.session_state["active_session"] = session_ids[0]
        else:
            st.session_state["active_session"] = create_new_session()
            st.rerun()

    # The Selection Widget
    # We map the display title back to the ID
    current_index = 0
    if st.session_state["active_session"] in session_ids:
        current_index = session_ids.index(st.session_state["active_session"])

    selected_title = st.radio(
        "History",
        session_titles,
        index=current_index,
        label_visibility="collapsed"
    )

    # Update active session based on selection
    # (Find the ID that matches the selected title)
    # Note: This is a simple lookup; assumes unique titles for simplicity in this demo.
    # A more robust app would use a custom component, but this works for local use.
    for sid, info in sessions.items():
        if info["title"] == selected_title:
            st.session_state["active_session"] = sid
            break

# --- 4. MAIN CHAT INTERFACE ---
active_id = st.session_state["active_session"]
messages = load_session(active_id)

st.header(f"💬 {sessions[active_id]['title']}")

# Add a RAG Toggle to the Sidebar
with st.sidebar:
    st.divider()
    use_rag = st.checkbox("🔍 Enable RAG (Search Docs)", value=True)

# Display History
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Handling
if user_input := st.chat_input("Ask about code or datasheets..."):
    # Add User Message
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    save_session(active_id, messages)

    # Context Variables
    context_text = ""
    sources = set()

    # RAG LOGIC (Only runs if Toggle is ON)
    if use_rag:
        with st.spinner("🔍 Checking docs..."):
            query_vec = embedder.encode([user_input]).tolist()
            # We added a 'distance' check here to filter bad matches
            results = collection.query(query_embeddings=query_vec, n_results=3)

            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    # Chroma returns distance (lower is better).
                    # A distance > 1.5 usually means "I found nothing relevant, just noise."
                    # You can uncomment this if you want strict filtering:
                    # if results['distances'][0][i] > 1.5: continue

                    meta = results['metadatas'][0][i]
                    src_name = os.path.basename(meta['source'])
                    context_text += f"-- SOURCE: {src_name} --\n{doc}\n\n"
                    sources.add(src_name)

    # Generation
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # DYNAMIC PROMPT: Changes based on whether RAG was used
        if context_text:
            system_instruction = (
                "You are an expert developer. Use the provided Context to answer. "
                "If the context is irrelevant, ignore it and use your own knowledge."
            )
            final_prompt = f"{system_instruction}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{user_input}"
        else:
            # Fallback to standard chat if RAG is off or found nothing
            final_prompt = user_input

        stream = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': final_prompt}],
            stream=True
        )

        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            message_placeholder.markdown(full_response + "▌")

        if sources:
            footer = "\n\n**📚 Sources:** " + ", ".join([f"`{s}`" for s in sources])
            full_response += footer
            message_placeholder.markdown(full_response)

    # Save Assistant Response
    messages.append({"role": "assistant", "content": full_response})
    save_session(active_id, messages)

    if len(messages) == 2:
        st.rerun()