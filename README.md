
# RAG-Fusion AI Chatbot

## Overview
RAG-Fusion AI Chatbot is a full-stack **Retrieval-Augmented Generation (RAG)** application that combines real-time web search, contextual knowledge retrieval, and generative AI to deliver intelligent and factual responses.

It integrates **FastAPI**, **MongoDB**, and **Google Gemini 2.5 Pro** on the backend, and an interactive **Streamlit** interface on the frontend, allowing users to chat, revisit past sessions, and explore AI-generated insights.

---

## Features
- **Real-Time Web Search**: Uses the Serper API to fetch the latest, relevant web data.
- **Context-Aware Responses**: Combines retrieved content with Gemini AI for factual answers.
- **MongoDB Integration**: Stores chat history and extracted documents for reuse.
- **Knowledge Graph Visualization**: Extracts entities and relationships using SpaCy and NetworkX.
- **Persistent Sessions**: Allows users to revisit previous chats with stored context.
- **Interactive UI**: Streamlit-powered frontend for seamless user experience.

---

## Technologies Used

### Backend
- **Framework**: FastAPI  
- **Database**: MongoDB (with Motor for asynchronous operations)  
- **AI Model**: Google Gemini 2.5 Pro  
- **Retrieval Engine**: LangChain + FAISS  
- **Search API**: Serper API (Google Search)  
- **NLP Tools**: SpaCy, Sentence Transformers, BeautifulSoup  

### Frontend
- **Framework**: Streamlit  
- **Communication**: REST API calls using Requests  
- **UI Features**:
  - Sidebar for chat sessions  
  - Real-time updates  
  - Message display with source URLs  

### Tools and Libraries
- FastAPI  
- Streamlit  
- LangChain  
- Google Generative AI SDK  
- Motor (Async MongoDB)  
- Pydantic  
- BeautifulSoup4  
- FAISS  
- NetworkX  
- SpaCy  
- Python-dotenv  

---

## Installation

### Prerequisites
- Python 3.9+  
- MongoDB installed and running locally or on MongoDB Atlas  
- API keys for:
  - Google Gemini  
  - Serper (https://serper.dev)

### Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/web-agent.git
cd web-agent
```

#### 2. Set Up a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS/Linux
```

#### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

#### 4. Create a `.env` File inside the `backend` folder
```bash
GEMINI_API_KEY=your_gemini_api_key
SERPER_API_KEY=your_serper_api_key
MONGO_URI=mongodb://localhost:27017/rag_db
```

#### 5. Start the MongoDB Server
```bash
mongod
```

#### 6. Run the Backend (FastAPI)
```bash
cd backend
uvicorn main:app --reload
```

#### 7. Run the Frontend (Streamlit)
```bash
cd ../frontend
streamlit run app.py
```

#### 8. Access the App
Open your browser and go to:  
```
http://localhost:8501
```

---

## Usage

1. Open the Streamlit interface in your browser.
2. Type your question in the chat box.
3. The system will:
   - Search the web for relevant pages.
   - Extract and analyze content.
   - Generate a concise, factual answer using Gemini AI.
4. The chatbot displays:
   - The generated answer
   - Related source URLs
   - Stored chat history for each session
5. You can revisit or continue any session from the sidebar.

---

## Project Structure
```bash
web-agent/
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── rag_core.py         # RAG pipeline (search, retrieval, generation)
│   ├── db.py               # MongoDB CRUD operations
│   ├── constants.py        # Environment variables loader
│   ├── requirements.txt    # Dependencies list
│   └── .env                # API keys and DB credentials
│
├── frontend/
│   ├── app.py              # Streamlit application (UI logic)
│
└── README.md               # Project documentation
```

---

## API Endpoints

| Endpoint           | Method | Description                                      |
|--------------------|--------|--------------------------------------------------|
| `/query`           | POST   | Processes a user query and returns response      |
| `/history/{chat_id}` | GET   | Retrieves chat history for a given session       |
| `/sessions`        | GET    | Lists all existing chat sessions                 |
| `/`                | GET    | Root endpoint (API status check)                 |

---

## Example

**Input**  
User: “What is the theory of relativity?”

**Output**
```yaml
Einstein’s theory of relativity describes how space and time are connected for objects in motion.
It introduced the concept that space and time are relative to the observer’s motion and that gravity affects both.
```

**Sources**  
- https://www.space.com/theory-of-relativity-explained  
- https://www.britannica.com/science/theory-of-relativity  

---

## Troubleshooting

| Issue                     | Possible Cause                  | Solution                                 |
|---------------------------|----------------------------------|------------------------------------------|
| 503: Model overloaded     | Gemini API temporarily busy      | Retry after a few seconds                |
| Chat history not showing  | Old schema without role field    | Use updated database structure           |
| Cannot connect to FastAPI | Backend not running              | Start `uvicorn main:app --reload`        |
| LF will be replaced by CRLF | Windows line ending warning    | Run `git config --global core.autocrlf true` |

---

## Future Enhancements
- User authentication and role-based access
- PDF and document ingestion for contextual retrieval
- Dashboard for usage analytics
- File upload and document-based Q&A
- Plugin system for custom tools and extensions
- Multi-language support

---

## Contributing

Contributions are welcome!  
To contribute:

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Submit a pull request  

---

## Author

**Manideep Palnati**  
BBA (AI & Data Science) 

---

## License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with proper credit.

---

## Acknowledgements

- **LangChain** — Text processing and retrieval utilities  
- **Google Gemini API** — AI text generation  
- **Serper API** — Real-time web search integration  
- **Streamlit** — Frontend framework  
- **FastAPI** — Backend framework  
- **MongoDB** — NoSQL database  

---
```

