from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import run_rag_query
from db import insert_chat_message, get_chat_history, get_all_chat_sessions
from typing import List, Dict

app = FastAPI(
    title="RAG-Fusion AI Agent API",
    description="An API for a retrieval-augmented generation agent using FastAPI, MongoDB, and Gemini.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    query: str
    chat_id: str

class QueryResponse(BaseModel):
    answer: str
    urls: List[str]
    context: str
    kg_nodes: int
    kg_edges: int

class ChatSession(BaseModel):
    chat_id: str
    title: str
    last_timestamp: str

class ChatHistoryResponse(BaseModel): #Represents the entire conversation history for a given session.
    history: List[Dict]

class ChatSessionsResponse(BaseModel): #Represents a list of all chat sessions — used by your Streamlit sidebar
    sessions: List[ChatSession]

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Receives a user query, processes it through the RAG pipeline, 
    and returns the generated answer and supporting information.
    """
    result = await run_rag_query(request.query)
    # Save the interaction to the database
    await insert_chat_message(request.query, result["answer"], request.chat_id, result.get("urls", []))
    return result

@app.get("/history/{chat_id}", response_model=ChatHistoryResponse)
async def fetch_history(chat_id: str):
    """
    Retrieves the chat history for a specific chat_id from the database.
    """
    history = await get_chat_history(chat_id)
    return {"history": history}

@app.get("/sessions", response_model=ChatSessionsResponse)
async def fetch_sessions():
    """
    Retrieves a list of all unique chat sessions.
    """
    sessions = await get_all_chat_sessions()
    return {"sessions": sessions}

@app.get("/") #A simple root endpoint that confirms the API is running.
async def root():
    return {"message": "Welcome to the RAG-Fusion AI Agent API!"}
