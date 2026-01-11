import streamlit as st
import ollama
import json
import os

HISTORY_FILE = "chat_history.json"

st.title("📻 GNU Radio Local Agent (Persistent)")

# --- FUNCTION: Load History from Disk ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return [] # Return empty list if no file exists

# --- FUNCTION: Save History to Disk ---
def save_history(messages):
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f)

# 1. Initialize State (Load from file on first run)
if "messages" not in st.session_state:
    st.session_state["messages"] = load_history()

# --- SIDEBAR: Controls ---
with st.sidebar:
    st.header("Memory Management")
    if st.button("🗑️ Clear History"):
        st.session_state["messages"] = []
        save_history([]) # Wipe the file too
        st.rerun() # Refresh the app

# 2. Display previous history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. The Input Box
if user_input := st.chat_input("Ask about GNU Radio..."):
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Save immediately (in case you crash)
    save_history(st.session_state["messages"])

    # 4. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Call your LOCAL model
        stream = ollama.chat(
            model='gnuradio-coder', 
            messages=st.session_state["messages"], 
            stream=True
        )
        
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
    
    # Save assistant message to history + File
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    save_history(st.session_state["messages"])
