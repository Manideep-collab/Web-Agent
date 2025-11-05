import motor.motor_asyncio
from constants import MONGO_URI
from typing import List, Dict
import datetime

# Initialize MongoDB client
try:
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    database = client.rag_db
    document_collection = database.rag_documents
    chat_collection = database.chat_history
    print("MongoDB client initialized.")
except Exception as e:
    print(f"Error initializing MongoDB client: {e}")
    # In a real application, this would be a critical failure. For now, we proceed.

async def insert_chat_message(query: str, answer: str, chat_id: str, urls: List[str] = None):
    """Inserts both user and assistant messages into the database."""
    now = datetime.datetime.now()
    urls = urls or []

    messages = [
        {
            "chat_id": chat_id,
            "role": "user",
            "query": query,
            "timestamp": now
        },
        {
            "chat_id": chat_id,
            "role": "assistant",
            "answer": answer,
            "urls": urls,
            "timestamp": now
        }
    ]
    try:
        await chat_collection.insert_many(messages)
    except Exception as e:
        print(f"Error inserting chat messages: {e}")


async def get_chat_history(chat_id: str) -> List[Dict]:
    """Retrieves the chat messages for a specific chat_id."""
    history = []
    try:
        cursor = chat_collection.find({"chat_id": chat_id}).sort("timestamp", 1)
        async for document in cursor:
            document["_id"] = str(document["_id"])
            history.append(document)
    except Exception as e:
        print(f"Error retrieving chat history for {chat_id}: {e}")
    return history

async def get_all_chat_sessions() -> List[Dict]:
    """Retrieves a list of all unique chat sessions with their first message."""
    sessions = []
    try:
        # Aggregate to find the first message of each unique chat_id
        pipeline = [
            {"$sort": {"timestamp": 1}}, #Sorts all chat messages in chronological order (oldest → newest).
            {"$group": {
                "_id": "$chat_id",
                "first_query": {"$first": "$query"},
                "last_timestamp": {"$last": "$timestamp"}
            }},
            {"$sort": {"last_timestamp": -1}}
        ]
        async for doc in chat_collection.aggregate(pipeline):
            sessions.append({
                "chat_id": doc["_id"],
                "title": doc["first_query"],
                "last_timestamp": doc["last_timestamp"].isoformat()
            })
    except Exception as e:
        print(f"Error retrieving chat sessions: {e}")
    return sessions

async def insert_rag_document(query: str, url: str, content: str, embedding: List[float]):
    """Inserts a new RAG document into the database."""
    document = {
        "query": query,
        "url": url,
        "content": content,
        "embedding": embedding,
        "timestamp": datetime.datetime.now()
    }
    try:
        await document_collection.insert_one(document) #without await the call wont excevute properly leading to incomplete data
    except Exception as e:
        print(f"Error inserting RAG document: {e}")

async def find_similar_documents(query_embedding: List[float], limit: int = 5) -> List[Dict]:
    """Finds similar RAG documents using vector search (requires MongoDB Atlas Vector Search).
    For a simple implementation, we will just return the most recent documents for now,
    as a full vector search setup is complex. The user will need to implement a proper
    vector search index on their MongoDB Atlas cluster.
    To retrieve conceptually similar documents, not necessarily identical queries
    """
    # NOTE: This is a placeholder for a proper vector search.
    # A real implementation would use $vectorSearch aggregation stage.
    # For now, we will just retrieve the most recent documents related to the query.
    # A better, but still simple, approach is to search by query text.
    history = [] #uses regex text search to find similar documents. temporary stand-in for true embedding based retrieval
    try:
        # Simple text search on the query field
        cursor = document_collection.find({"query": {"$regex": query, "$options": "i"}}).sort("timestamp", -1).limit(limit)
        async for document in cursor:
            document["_id"] = str(document["_id"])
            history.append(document)
    except Exception as e:
        print(f"Error retrieving similar documents: {e}")
    return history

async def check_document_exists(url: str) -> bool:
    """Checks if a document from a given URL already exists in the database."""
    try:
        count = await document_collection.count_documents({"url": url})
        return count > 0
    except Exception as e:
        print(f"Error checking document existence: {e}")
        return False

async def get_documents_by_query(query: str, limit: int = 6) -> List[Dict]: 
    # To avoid redundant web searches and scraping for repeated or similar user queries.
    # “Has this query (or a very similar one) already been asked before?
    # If yes, fetch the same documents from the database instead of re-scraping.”
    """Retrieves documents previously stored for a specific query."""
    documents = []
    try:
        # Find documents where the stored query is similar to the new query
        cursor = document_collection.find({"query": {"$regex": query, "$options": "i"}}).sort("timestamp", -1).limit(limit)
        async for document in cursor:
            document["_id"] = str(document["_id"])
            documents.append(document)
    except Exception as e:
        print(f"Error retrieving documents by query: {e}")
    return documents

async def get_all_documents() -> List[Dict]:
    """Retrieves all RAG documents."""
    documents = []
    try:
        cursor = document_collection.find().sort("timestamp", -1)
        async for document in cursor:
            document["_id"] = str(document["_id"])
            documents.append(document)
    except Exception as e:
        print(f"Error retrieving all documents: {e}")
    return documents

async def delete_all_documents():
    """Deletes all RAG documents."""
    try:
        await document_collection.delete_many({})
    except Exception as e:
        print(f"Error deleting all documents: {e}")

    """Inserts a new chat message into the database."""
    message = {
        "query": query,
        "answer": answer,
        "timestamp": datetime.datetime.now()
    }
    try:
        await chat_collection.insert_one(message)
    except Exception as e:
        print(f"Error inserting chat message: {e}")

