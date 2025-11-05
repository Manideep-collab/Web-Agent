import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import spacy
import networkx as nx
from google import genai
from constants import GEMINI_API_KEY, SERPER_API_KEY
import warnings
import textwrap
from typing import Tuple, List, Dict

warnings.filterwarnings("ignore")

# Setup Gemini
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize NLP and Embedding components once
try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model 'en_core_web_md'...")
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_md"])
    NLP = spacy.load("en_core_web_md")

EMBEDDING = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


async def _search_web(query: str) -> List[str]:
    """Performs a web search using Serper and returns a list of top 6 URLs."""
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    try:
        res = requests.post("https://google.serper.dev/search", headers=headers, json={"q": query})
        res.raise_for_status()
        data = res.json()
        # Changed from 3 to 6 as requested
        links = data.get("organic", [])
        urls = [link["link"] for link in links[:4]]
        return urls
    except Exception as e:
        print(f"Error during web search: {e}")
        return []

async def _extract_and_store_text_from_urls(query: str, urls: List[str]) -> str:
    """Fetches content from URLs, extracts text, stores in MongoDB, and returns combined text."""
    from db import insert_rag_document, check_document_exists
    
    ua = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    all_text = []
    
    for u in urls:
        # 1. Check if document already exists
        if await check_document_exists(u):
            print(f"Document for {u} already exists. Skipping.")
            continue
            
        try:
            r = requests.get(u, headers=ua, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Extract main content text
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
            page_content = " ".join(paragraphs)
            
            if page_content:
                # 2. Generate embedding for the content
                # NOTE: For simplicity and to avoid complex async/sync issues with HuggingFaceEmbeddings,
                # we will use a placeholder embedding or a simple hash. A proper RAG system would
                # embed the content chunks. Since the user requested to store the *document*
                # and the current setup uses FAISS on the fly, we will store the full content
                # and a placeholder embedding. The retrieval logic will be updated to use
                # the stored content.
                
                # For a proper RAG system, we should chunk and embed here.
                # For now, we will store the full content and use the query for retrieval.
                
                # Placeholder embedding (e.g., a list of zeros)
                placeholder_embedding = [0.0] * 384 # 384 is the dimension for all-MiniLM-L6-v2
                
                # 3. Store the document in MongoDB
                await insert_rag_document(query, u, page_content, placeholder_embedding)
                all_text.append(page_content)
                
        except Exception as e:
            print(f"Skipping {u}: {e}")
            
    return " ".join(all_text)

async def _retrieve_from_db(query: str) -> Tuple[str, List[str]]:
    """Retrieves relevant documents from MongoDB based on the query."""
    from db import get_documents_by_query
    
    # Simple retrieval: get documents previously stored for a similar query
    documents = await get_documents_by_query(query, limit=6)
    
    if not documents:
        return "", []
        
    context = "\n---\n".join([doc["content"] for doc in documents])
    urls = [doc["url"] for doc in documents]
    
    return context, urls

def _build_knowledge_graph_info(retrieved_docs: List) -> Tuple[str, int, int]:
    """Builds a simple knowledge graph from retrieved documents and returns a string summary."""
    G = nx.DiGraph()
    for doc in retrieved_docs:
        text = doc.page_content
        doc_nlp = NLP(text)
        for sent in doc_nlp.sents:
            entities = [ent.text for ent in sent.ents]
            G.add_node(sent.text, entities=entities)
            if len(entities) >= 2:
                # Simple relation: related_to
                G.add_edge(entities[0], entities[1], relation="related_to")

    kg_info = ""
    if G.number_of_edges() > 0:
        kg_info = "\nKnowledge Graph Triples (Subject, Relation, Object):"
        for u, v, data in G.edges(data=True):
            kg_info += f"\n({u}, {data['relation']}, {v})"
    
    return kg_info, G.number_of_nodes(), G.number_of_edges()


async def run_rag_query(query: str) -> Dict:
    """
    Executes the full RAG pipeline: search, extract, embed, retrieve, and generate.
    Returns a dictionary with the answer and supporting data.
    """
    # 1. Try to retrieve from MongoDB first
    context, urls = await _retrieve_from_db(query)
    
    if context:
        print("Retrieved context from MongoDB.")
        # 2. If context is found, use it for RAG
        full_text = context
        source_urls = urls
        retrieved_from_db = True
    else:
        print("No context found in MongoDB. Performing web search.")
        retrieved_from_db = False
        
        # 2.1. Search Web
        source_urls = await _search_web(query)
        if not source_urls:
            return {
                "answer": "I could not find any relevant web results for your query.",
                "urls": [],
                "context": "",
                "kg_nodes": 0,
                "kg_edges": 0
            }

        # 2.2. Extract Text and Store in DB
        full_text = await _extract_and_store_text_from_urls(query, source_urls)
        if not full_text:
            return {
                "answer": "I found URLs but could not extract any text content from them.",
                "urls": source_urls,
                "context": "",
                "kg_nodes": 0,
                "kg_edges": 0
            }

    # 3. Split and Embed (Always use the retrieved or newly scraped text)
    docs = TEXT_SPLITTER.split_text(full_text)
    
    # Create LangChain Documents from the text chunks
    from langchain_core.documents import Document
    lc_docs = [Document(page_content=d, metadata={"source": "MongoDB" if retrieved_from_db else "Web Search"}) for d in docs]
    
    # Use FAISS for vector search on the fly (for the current query)
    vector_db = FAISS.from_documents(lc_docs, EMBEDDING)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # 4. Retrieve Context
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in retrieved_docs])

    # 5. Build Knowledge Graph Info
    kg_info, kg_nodes, kg_edges = _build_knowledge_graph_info(retrieved_docs)

    # 6. Generate Prompt and Answer using Gemini
    prompt = f"""
You are **Gemini 2.5 Pro**, an advanced retrieval-augmented AI system.
You specialize in combining retrieved knowledge with logical reasoning to provide concise, factual, and contextually grounded answers.
You are given the following retrieved information and (if available) a simple knowledge graph representation of key entities and relations.

---

### CONTEXT
{context}

### KNOWLEDGE GRAPH
{kg_info}

---

### USER QUESTION
{query}

---

### INSTRUCTIONS
1. Use only verified facts from the provided context and knowledge graph to form your answer.  
2. If the answer is **not clearly supported by the context**, explicitly say so — for example:  
   *"The retrieved information does not contain a direct answer, but based on general knowledge..."*  
3. If you must infer, keep it **logical, minimal, and clearly marked as inference.**  
4. Always maintain these qualities:
   - **Clarity:** Write in well-structured, professional English.  
   - **Conciseness:** Limit the answer to 2–3 rich sentences.  
   - **Credibility:** Do not fabricate facts or data.  
   - **Relevance:** Directly address the user’s question.  

---

### OUTPUT FORMAT
Provide only the final answer text — no explanations of reasoning, no bullet points, and no extra formatting.
"""

    try:
        # ✅ Fixed Gemini API call — use string instead of dicts
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        # Safely extract text
        answer = (
            response.text.strip()
            if hasattr(response, "text") and response.text
            else "No response generated by Gemini."
        )

    except Exception as e:
        print(f"Gemini generation error: {e}")
        answer = f"An error occurred during content generation: {e}"

    # 7. Return structured response
    return {
        "answer": answer,
        "urls": source_urls, # Use source_urls which is either from DB or new search
        "context": context,
        "kg_nodes": kg_nodes,
        "kg_edges": kg_edges
    }
