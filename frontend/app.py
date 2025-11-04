import streamlit as st
import requests
import datetime
import json
import os


FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chat Assistant", layout="wide")


def get_sessions():
    try:
        response = requests.get(f"{FASTAPI_URL}/sessions")
        response.raise_for_status()
        return response.json().get("sessions", [])
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")
        return []

def get_history(chat_id: str):
    try:
        response = requests.get(f"{FASTAPI_URL}/history/{chat_id}")
        response.raise_for_status()
        return response.json().get("history", [])
    except Exception as e:
        st.error(f"Error fetching history for {chat_id}: {e}")
        return []


def send_query(query: str, chat_id: str):
    try:
        response = requests.post(
            f"{FASTAPI_URL}/query",
            json={"query": query, "chat_id": chat_id},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI backend. Please ensure it’s running.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None


if "sessions" not in st.session_state:
    st.session_state.sessions = get_sessions()

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def load_active_session(chat_id):
    st.session_state.active_chat_id = chat_id
    st.session_state.messages = get_history(chat_id)
    st.session_state.sessions = get_sessions() # Refresh sessions list
    st.rerun()


st.sidebar.title("💬 Chat History")

# List all sessions
if st.session_state.sessions:
    for session in st.session_state.sessions:
        title = session["title"][:40] + "..." if len(session["title"]) > 40 else session["title"]
        if st.sidebar.button(title, key=session["chat_id"]):
            load_active_session(session["chat_id"])
else:
    st.sidebar.info("No previous chats yet.")

# New chat button
if st.sidebar.button("➕ New Chat"):
    import uuid
    new_id = str(uuid.uuid4())
    load_active_session(new_id) # This will set active_chat_id and clear messages
    st.rerun()


st.title("🧠 RAG-Fusion AI Chatbot")

# Display existing messages
for msg in st.session_state.messages:
    role = msg.get("role", None)

    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg.get("query", ""))
    elif role == "assistant":
        with st.chat_message("assistant"):
            answer = msg.get("answer", "")
            urls = msg.get("urls", [])
            response_content = f"{answer}\n\n---\n\n**Sources:**\n"
            if urls:
                for i, url in enumerate(urls):
                    response_content += f"- [{url}]({url})\n"
            else:
                response_content += "No sources found."
            st.markdown(response_content)


# Chat input box
if prompt := st.chat_input("Ask something..."):
    # Ensure a chat session exists
    if not st.session_state.active_chat_id:
        import uuid
        st.session_state.active_chat_id = str(uuid.uuid4())

    # Display user message
    user_msg = {
        "role": "user",
        "query": prompt,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get backend response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = send_query(prompt, st.session_state.active_chat_id)
            if result:
                answer = result.get("answer", "No valid answer found.")
                urls = result.get("urls", [])
                
                # Format the assistant's response
                response_content = f"{answer}\n\n---\n\n**Sources:**\n"
                if urls:
                    for i, url in enumerate(urls):
                        response_content += f"- [{url}]({url})\n"
                else:
                    response_content += "No sources found."
                    
                st.markdown(response_content)

                bot_msg = {
                    "role": "assistant",
                    "answer": answer,
                    "urls": urls,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state.messages.append(bot_msg)
            else:
                st.error("Failed to get a response from the backend.")

    # Refresh sessions list and rerun to update UI
    st.session_state.sessions = get_sessions()
    st.rerun()
